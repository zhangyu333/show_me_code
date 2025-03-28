//
// Created by 张宇 on 2025/3/25.
//

#ifndef WAV2LIP_CPP_LMSDET_HPP
#define WAV2LIP_CPP_LMSDET_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class DetLMS {
public:
    explicit DetLMS(const std::string &model_path);

    ~DetLMS() = default;

    cv::Mat call(cv::Mat &frame);

    cv::Mat call(const std::string &image_path);

private:
    cv::Mat input_image;

    int org_height = {}, org_weight = {};
    float height_ratio = {}, weight_ratio = {};

    std::vector<Ort::Value> output_tensors;
    std::vector<int64_t> input_shape_{1, 3, 112, 112};
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::SessionOptions session_options = Ort::SessionOptions();
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"output1", "output"};
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DETLMS");
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);

    void preProc(cv::Mat &image);

    void inference();

    cv::Mat postProc();
};

DetLMS::DetLMS(const std::string &model_path) {
    this->session_options.SetIntraOpNumThreads(1);
    this->session_options.SetInterOpNumThreads(1);
    this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
}

cv::Mat DetLMS::call(cv::Mat &frame) {
    this->preProc(frame);
    this->inference();
    return this->postProc();
}

cv::Mat DetLMS::call(const std::string &image_path) {
    auto image = cv::imread(image_path);
    return this->call(image);
}

void DetLMS::preProc(cv::Mat &image) {
    this->org_height = image.rows;
    this->org_weight = image.cols;
    this->height_ratio = static_cast<float>(this->org_height) / 112;
    this->weight_ratio = static_cast<float>(this->org_weight) / 112;
    image.convertTo(this->input_image, CV_32FC3);
    cv::resize(this->input_image, this->input_image, cv::Size(112, 112));
    this->input_image /= cv::Scalar(255., 255., 255.);
    this->input_image = cv::dnn::blobFromImage(this->input_image);
}

void DetLMS::inference() {
    std::vector<float> input_image_vec;
    input_image_vec.resize(3 * 112 * 112);

    float *input_1 = input_image_vec.data();
    ::memcpy(input_1, this->input_image.data, 112 * 112 * 3 * sizeof(float));

    auto input_tensor = Ort::Value::CreateTensor<float>(
            this->memory_info,
            input_1,
            input_image_vec.size(),
            input_shape_.data(),
            input_shape_.size());

    clock_t st = clock();
    this->output_tensors = this->session->Run(Ort::RunOptions{nullptr},
                                              this->input_node_names.data(),
                                              &input_tensor,
                                              this->input_node_names.size(),
                                              this->output_node_names.data(),
                                              this->output_node_names.size()
    );
    clock_t et = clock();
    printf("LMS检测推理耗时: %fms\n", ((double) (et - st) / CLOCKS_PER_SEC) * 1000);
}

cv::Mat DetLMS::postProc() {
    auto outputs = this->output_tensors[1].GetTensorMutableData<float>();
    cv::Mat lms = cv::Mat::ones(cv::Size(2, 106), CV_32F);
    ::memcpy(lms.data, outputs, 2 * 106 * sizeof(float));
    lms(cv::Range::all(), cv::Range(0, 1)) *= 112;
    lms(cv::Range::all(), cv::Range(1, 2)) *= 112;
    lms(cv::Range::all(), cv::Range(0, 1)) *= this->weight_ratio;
    lms(cv::Range::all(), cv::Range(1, 2)) *= this->height_ratio;
    return lms;
}

#endif //WAV2LIP_CPP_LMSDET_HPP
