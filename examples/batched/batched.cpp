#include "common.h"
#include "llama.h"
#include "batched.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

using json = nlohmann::json;


int Sampler::estimate_max_length(int prompt_tokens, int token_limit, int token_estimate) {
    if (token_estimate != -1) {
        return std::min(token_estimate, token_limit);
    }
    return std::min(static_cast<int>(2.15 * prompt_tokens), token_limit);
}

llama_batch Sampler::configure_batch(int num_samples, std::vector<llama_token> &tokens, llama_context *ctx) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) tokens.size());

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        throw std::runtime_error("llama decode failed");
    }

    // assign the system KV cache to all parallel sequences
    // this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    for (int32_t i = 1; i < num_samples; ++i) {
        llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens);
    }

    return batch;
}

int Sampler::calculate_key_value_cache(int num_tokens_used, int max_tokens, int num_samples) {
    return num_tokens_used + (max_tokens - num_tokens_used) * num_samples;
}

std::string Sampler::format_prompt(const std::string &python_code) {
    std::stringstream prompt;
    prompt
            << "[INST] You are a tool that converts Python code to functionally equivalent Java code, adhering to the following constraints: \n"
            << "(1) Your response must be enclosed between the [JAVA] and [/JAVA] tags should only contain Java code. \n"
            << "(2) Your code should be executable and should include all necessary imports. \n"
            << "(3) Your solution should be encapsulated within a class named 'Solution' that includes a public static void main(String[] args) method. \n"
            << "(4) The Python code may include advanced features such as list comprehensions, lambda functions, and generators. Your translation should express these features idiomatically to Java. \n\n"
            << "[PYTHON]\n"
            << python_code
            << "\n[/PYTHON]\n"
            << "[/INST]";
    return prompt.str();
}

llama_token
Sampler::sample_candidate_token(llama_context *ctx, std::vector<llama_token_data> &candidates, int top_k, float top_p,
                                float temp) {
    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
    llama_sample_top_k(ctx, &candidates_p, top_k, 1);
    llama_sample_top_p(ctx, &candidates_p, top_p, 1);
    llama_sample_temp(ctx, &candidates_p, temp);
    llama_token new_token_id = llama_sample_token(ctx, &candidates_p);
    return new_token_id;
}

llama_model *Sampler::load_model(int gpu_layers, std::string &path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    static llama_model *model = llama_load_model_from_file(path.c_str(), model_params);

    if (model == nullptr) {
        throw std::runtime_error("error: unable to load model from " + path);
    }

    return model;
}

std::vector<std::string> Sampler::generate_streams(GenerateStreamsParams &params) {
    std::vector<std::string> streams(params.num_samples);
    std::vector<int32_t> i_batch(params.num_samples, params.batch.n_tokens - 1);

    const auto t_main_start = ggml_time_us();

    int n_current = params.batch.n_tokens;
    int n_decoded = 0;

    while (n_current < params.max_tokens) {
        llama_batch_clear(params.batch);
        for (int32_t i = 0; i < params.num_samples; ++i) {
            if (i_batch[i] < 0) {
                continue;
            }

            auto n_vocab = llama_n_vocab(params.model);
            auto *logits = llama_get_logits_ith(params.context, i_batch[i]);
            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            const llama_token new_token_id = this->sample_candidate_token(params.context, candidates, params.top_k,
                                                                          params.top_p, params.temp);

            if (new_token_id == llama_token_eos(params.model) || n_current == params.max_tokens) {
                i_batch[i] = -1;
                LOG_TEE("%s: stream %d finished at n_cur = %d", __func__, i, n_current);
                continue;
            }

            streams[i] += llama_token_to_piece(params.context, new_token_id);
            i_batch[i] = params.batch.n_tokens;
            llama_batch_add(params.batch, new_token_id, n_current, {i}, true);
            n_decoded += 1;
        }

        if (params.batch.n_tokens == 0) {
            break;
        }

        n_current += 1;

        if (llama_decode(params.context, params.batch)) {
            throw std::runtime_error("Error to evaluate");
        }
    }
    const auto t_main_end = ggml_time_us();
    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decoded, (t_main_end - t_main_start) / 1000000.0f,
            n_decoded / ((t_main_end - t_main_start) / 1000000.0f));
    return streams;
}


json Sampler::process_batch(BatchProcessParams &params) {
    llama_backend_init(false);
    llama_model *model = this->load_model(params.gpu_layers, params.model_path);
    std::string prompt = this->format_prompt(params.python_code);

    LOG_TEE("%s: Prompt: %s\n", __func__, prompt.c_str());

    std::vector<llama_token> tokens;
    tokens = ::llama_tokenize(model, prompt, true);
    int max_tokens = this->estimate_max_length(params.prompt_context_size, params.upper_token_limit,
                                               params.token_estimate);

    LOG_TEE("%s: Max Tokens Estimate: %d\n", __func__, max_tokens);
    int cache_size = this->calculate_key_value_cache(tokens.size(), max_tokens, params.num_samples);

    LOG_TEE("%s: Cache Size Estimate: %d\n", __func__, cache_size);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cache_size;
    ctx_params.n_batch = std::max(cache_size, params.num_samples);
    ctx_params.n_threads = params.batch_threads;
    ctx_params.n_threads_batch = params.batch_threads;

    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == nullptr) {
        throw std::runtime_error("error: failed to create the llama_context");
    }

    const int n_ctx = llama_n_ctx(ctx);

    if (cache_size > n_ctx) {
        throw std::runtime_error("error: the required KV cache is not big enough");
    }

    llama_batch batch = this->configure_batch(params.num_samples, tokens, ctx);

    GenerateStreamsParams generate_params = GenerateStreamsParams::fromBatchProcessParams(
            params, model, ctx, batch, max_tokens
    );

    std::vector<std::string> streams = this->generate_streams(generate_params);

    llama_batch_free(batch);
    llama_free_model(model);
    llama_backend_free();

    json output;
    output["max_tokens"] = max_tokens;
    output["streams"] = streams;

    return output;
}

int main() {
    try {
        // Read all lines from stdin into a string
        std::string input;
        for (std::string line; std::getline(std::cin, line);) {
            input += line + "\n";
        }

        // Parse the string into a json object
        json json_params = json::parse(input);
        auto params = json_params.template get<BatchProcessParams>();

        Sampler sampler;
        json output = sampler.process_batch(params);
        std::cout << output.dump() << std::endl;

    }
    catch (const json::parse_error &e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}
