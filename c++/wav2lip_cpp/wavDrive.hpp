//
// Created by 张宇 on 2025/3/26.
//

#ifndef WAV2LIP_CPP_WAVDRIVE_HPP
#define WAV2LIP_CPP_WAVDRIVE_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>

class WavDrive {
public:
    explicit WavDrive(const std::string &model_path);

    ~WavDrive() = default;

    void call(
            Eigen::Tensor<float, 4> &batch_mel_tensors,
            Eigen::Tensor<float, 4> &batch_image_tensors
    );

private:
    void inference(Eigen::Tensor<float, 4> &batch_mel_tensors,
                   Eigen::Tensor<float, 4> &batch_image_tensors);

    void postProc();

    std::vector<Ort::Value> output_tensors;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::SessionOptions session_options = Ort::SessionOptions();
    std::vector<const char *> input_node_names = {"audio_sequences", "face_sequences"};
    std::vector<const char *> output_node_names = {"outputs"};
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "WavDriver");
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);
    int batch_size = {};
};

void WavDrive::call(Eigen::Tensor<float, 4> &batch_mel_tensors, Eigen::Tensor<float, 4> &batch_image_tensors) {
    this->inference(batch_mel_tensors, batch_image_tensors);
    this->postProc();
}

WavDrive::WavDrive(const std::string &model_path) {
    this->session_options.SetIntraOpNumThreads(1);
    this->session_options.SetInterOpNumThreads(1);
    this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
}

void WavDrive::inference(Eigen::Tensor<float, 4> &batch_mel_tensors, Eigen::Tensor<float, 4> &batch_image_tensors) {
    this->batch_size = static_cast<int>(batch_mel_tensors.dimensions()[0]);
    auto batch_mel_tensor_dims = batch_mel_tensors.dimensions();
    auto batch_image_tensor_dims = batch_image_tensors.dimensions();

    std::vector<int64_t> batch_mel_tensors_shape{
            this->batch_size, batch_mel_tensor_dims[1], batch_mel_tensor_dims[2], batch_mel_tensor_dims[3]};
    std::vector<float> input_mel_vec;
    input_mel_vec.resize(
            this->batch_size * batch_mel_tensor_dims[1] * batch_mel_tensor_dims[2] * batch_mel_tensor_dims[3]);
    float *input_1 = input_mel_vec.data();
    ::memcpy(input_1, batch_mel_tensors.data(),
             this->batch_size * batch_mel_tensor_dims[1] * batch_mel_tensor_dims[2] * batch_mel_tensor_dims[3] *
             sizeof(float));

    std::vector<int64_t> batch_image_tensors_shape{this->batch_size, batch_image_tensor_dims[1],
                                                   batch_image_tensor_dims[2], batch_image_tensor_dims[3]};
    std::vector<float> input_image_vec;
    input_image_vec.resize(
            this->batch_size * batch_image_tensor_dims[1] * batch_image_tensor_dims[2] * batch_image_tensor_dims[3]);
    float *input_2 = input_image_vec.data();
    ::memcpy(input_2, batch_image_tensors.data(),
             this->batch_size * batch_image_tensor_dims[1] * batch_image_tensor_dims[2] * batch_image_tensor_dims[3] *
             sizeof(float));

    auto input_tensor1 = Ort::Value::CreateTensor<float>(
            this->memory_info,
            input_1,
            input_mel_vec.size(),
            batch_mel_tensors_shape.data(),
            batch_mel_tensors_shape.size());

    auto input_tensor2 = Ort::Value::CreateTensor<float>(
            this->memory_info,
            input_2,
            input_image_vec.size(),
            batch_image_tensors_shape.data(),
            batch_image_tensors_shape.size());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor1));
    ort_inputs.push_back(std::move(input_tensor2));
    clock_t st = clock();
    this->output_tensors = this->session->Run(
            Ort::RunOptions{nullptr},
            this->input_node_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            this->output_node_names.data(),
            this->output_node_names.size()
    );
    clock_t et = clock();
    printf("WavDrive推理耗时: %fms\n", ((double) (et - st) / CLOCKS_PER_SEC) * 1000);
}

void WavDrive::postProc() {
    auto outputs = this->output_tensors[0].GetTensorMutableData<float>();
    Eigen::Tensor<float, 4> output_tensor = {this->batch_size, 3, 96, 96};
    ::memcpy(output_tensor.data(), outputs, sizeof(float) * this->batch_size * 3 * 96 * 96);
    std::vector<cv::Mat> pred_faces(this->batch_size);

    for (size_t i = 0; i < this->batch_size; ++i) {
        Eigen::Tensor<float, 3> slice = output_tensor.chip(i, 0);
        cv::Mat face(96, 96, CV_32FC3);
        for (int y = 0; y < 96; ++y) {
            for (int x = 0; x < 96; ++x) {
                for (int c = 0; c < 3; ++c) {
                    face.at<cv::Vec3f>(y, x)[c] = slice(c, y, x);
                }
            }
        }
        face *= 255.;
        pred_faces[i] = face;
    }

}

#endif //WAV2LIP_CPP_WAVDRIVE_HPP
