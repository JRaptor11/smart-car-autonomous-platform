#include "tensorrt_runner.hpp"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
int volume(const nvinfer1::Dims& dims) {
    int total = 1;
    for (int i = 0; i < dims.nbDims; ++i) total *= dims.d[i];
    return total;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}
}

void TensorRTRunner::Logger::log(Severity severity, const char* msg) noexcept {
    if (!verbose_ && severity > Severity::kWARNING) return;
    std::cerr << "[TensorRT] " << msg << '\n';
}

TensorRTRunner::TensorRTRunner(const std::string& engine_path, bool verbose)
    : logger_(verbose) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine: " + engine_path);

    file.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> blob(size);
    file.read(blob.data(), static_cast<std::streamsize>(size));

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime.");

    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT engine.");

    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("Failed to create TensorRT execution context.");

    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) input_index_ = i;
        else output_index_ = i;
    }
    if (input_index_ < 0 || output_index_ < 0) {
        throw std::runtime_error("Expected one input binding and one output binding.");
    }

    const auto input_dims = engine_->getBindingDimensions(input_index_);
    const auto output_dims = engine_->getBindingDimensions(output_index_);
    if (input_dims.nbDims != 4) throw std::runtime_error("Expected NCHW input tensor.");

    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];
    input_count_ = volume(input_dims);
    output_count_ = volume(output_dims);

    check_cuda(cudaMalloc(&device_input_, sizeof(float) * input_count_), "cudaMalloc input");
    check_cuda(cudaMalloc(&device_output_, sizeof(float) * output_count_), "cudaMalloc output");
    bindings_[input_index_] = device_input_;
    bindings_[output_index_] = device_output_;
}

TensorRTRunner::~TensorRTRunner() {
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
}

std::vector<float> TensorRTRunner::infer(const std::vector<float>& input) {
    if (static_cast<int>(input.size()) != input_count_) {
        throw std::runtime_error("Unexpected TensorRT input size.");
    }
    check_cuda(cudaMemcpy(device_input_, input.data(), sizeof(float) * input_count_, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    if (!context_->enqueueV2(bindings_, 0, nullptr)) {
        throw std::runtime_error("TensorRT enqueueV2 failed.");
    }
    std::vector<float> output(output_count_);
    check_cuda(cudaMemcpy(output.data(), device_output_, sizeof(float) * output_count_, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return output;
}
