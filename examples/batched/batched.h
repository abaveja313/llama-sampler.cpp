//
// Created by Amrit Baveja on 1/13/24.
//

#ifndef LLAMA_CPP_BATCHED_H
#define LLAMA_CPP_BATCHED_H

#include <string>
#include <vector>
#include "llama.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;


struct BatchProcessParams {
    std::string python_code;
    std::string model_path;
    int gpu_layers;
    int batch_threads;
    int prompt_context_size;
    int upper_token_limit;
    int top_k;
    float top_p;
    float temp;
    int num_samples;
    bool use_limit;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(BatchProcessParams, python_code, model_path, gpu_layers, batch_threads, prompt_context_size,
            upper_token_limit, top_k, top_p, temp, num_samples, use_limit);
};

struct GenerateStreamsParams {
    llama_model *model;
    llama_context *context;
    llama_batch batch;
    int num_samples;
    int max_tokens;
    int top_k;
    float top_p;
    float temp;

    // Static builder function
    static GenerateStreamsParams fromBatchProcessParams(
            const BatchProcessParams &batchParams,
            llama_model *model,
            llama_context *context,
            llama_batch batch,
            int max_tokens
    ) {
        GenerateStreamsParams params;
        params.model = model;
        params.context = context;
        params.batch = batch;
        params.num_samples = batchParams.num_samples;
        params.max_tokens = max_tokens;
        params.top_k = batchParams.top_k;
        params.top_p = batchParams.top_p;
        params.temp = batchParams.temp;
        return params;
    }
};

class Sampler {
private:
    int estimate_max_length(int prompt_tokens, int token_limit, bool use_limit);

    llama_model load_model(int gpu_layers, std::string &path, llama_model &output);

    llama_token
    sample_candidate_token(llama_context *ctx, std::vector<llama_token_data> &candidates, int top_k, float top_p,
                           float temp);

    std::string format_prompt(const std::string &python_code);

    llama_model *load_model(int gpu_layers, std::string &path);

    llama_batch configure_batch(int num_samples, std::vector<llama_token> &tokens, llama_context *ctx);

    int calculate_key_value_cache(int num_tokens_used, int max_tokens, int num_samples);

    std::vector<std::string> generate_streams(GenerateStreamsParams &params);

public:
    json process_batch(BatchProcessParams &params);
};

#endif //LLAMA_CPP_BATCHED_H
