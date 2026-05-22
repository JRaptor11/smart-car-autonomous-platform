#pragma once

#include <NvInfer.h>

#include <string>
#include <vector>

class TensorRTRunner {
public:
    explicit TensorRTRunner(const std::string& engine_path, bool verbose = true);
    ~TensorRTRunner();

    TensorRTRunner(const TensorRTRunner&) = delete;
    TensorRTRunner& operator=(const TensorRTRunner&) = delete;

    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }

    std::vector<float> infer(const std::vector<float>& input);

private:
    class Logger : public nvinfer1::ILogger {
    public:
        explicit Logger(bool verbose) : verbose_(verbose) {}
        void log(Severity severity, const char* msg) noexcept override;
    private:
        bool verbose_;
    };

    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    int input_index_ = -1;
    int output_index_ = -1;
    int input_w_ = 0;
    int input_h_ = 0;
    int input_count_ = 0;
    int output_count_ = 0;

    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    void* bindings_[2] = {nullptr, nullptr};
};
