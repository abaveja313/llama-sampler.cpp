cmake_minimum_required(VERSION 3.14)  # Ensure you have a suitable minimum version

# Set the target
set(TARGET batched)
add_executable(${TARGET} batched.cpp batched.h)
include(FetchContent)

# Declare where to get nlohmann/json from
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.10.5  # Specify the commit hash or tag to use
)

# Make nlohmann/json available to the project
FetchContent_MakeAvailable(json)
# Link libraries
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} nlohmann_json::nlohmann_json)

# Specify C++ standard
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Install rules
install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
