//
// Created by 张宇 on 2024/6/24.
//

#ifndef GAZE_ESTIMATE_GAZE_ESTIMATE_HPP
#define GAZE_ESTIMATE_GAZE_ESTIMATE_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

typedef struct {
    std::vector<double> pitch_list;
    std::vector<double> yaw_list;
} PYRES;


class GazeEstimate {
public:
    explicit GazeEstimate(const std::string &model_path) {
        this->session_options.SetIntraOpNumThreads(1);
        this->session_options.SetInterOpNumThreads(1);
        this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        this->session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    };

    ~GazeEstimate() = default;

    PYRES call(std::vector<cv::Mat> &images);

private:
    cv::Mat blob, inp_mat;

    std::vector<double> pitch_list, yaw_list;

    void postProc();

    void inference();

    void preProc(std::vector<cv::Mat> &images);

    void preProc_(cv::Mat &image);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);
    std::shared_ptr<Ort::Session> session = nullptr;
    std::vector<const char *> input_node_names = {"inputs"};
    std::vector<const char *> output_node_names = {"gaze_pitchs", "gaze_yaws"};
    __attribute__((unused)) std::vector<Ort::Value> output_tensors;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "GAZE-ESTIMATE");
    Ort::SessionOptions session_options = Ort::SessionOptions();
};

PYRES GazeEstimate::call(std::vector<cv::Mat> &images) {
    clock_t st = clock();
    this->preProc(images);
    this->inference();
    printf("视线估计总耗时: %fms\n", ((double) (clock() - st) / CLOCKS_PER_SEC) * 1000);
    this->postProc();
    return PYRES{this->pitch_list, this->yaw_list};
}

void GazeEstimate::preProc(std::vector<cv::Mat> &images) {
    std::vector<cv::Mat> faces;
    for (auto &image: images) {
        preProc_(image);
        faces.emplace_back(this->inp_mat);
    }
    this->blob = cv::dnn::blobFromImages(faces);
}

void GazeEstimate::preProc_(cv::Mat &image) {
    cv::cvtColor(image, this->inp_mat, cv::COLOR_BGR2RGB);
    cv::resize(this->inp_mat, this->inp_mat, cv::Size(448, 448));
    this->inp_mat.convertTo(this->inp_mat, CV_32F);
    this->inp_mat /= cv::Scalar(255., 255., 255.);
    this->inp_mat -= cv::Scalar(0.485, 0.456, 0.406);
    this->inp_mat /= cv::Scalar(0.229, 0.224, 0.225);
}

cv::Mat softmax(const cv::Mat &src) {
    cv::Mat dst;
    auto max = *std::max_element(src.begin<float>(), src.end<float>());
    cv::exp((src - max), dst);
    auto sum = cv::sum(dst)[0];
    dst /= sum;
    return dst;
}


void GazeEstimate::postProc() {
    this->pitch_list.clear();
    this->yaw_list.clear();
    auto gaze_pitches_data = this->output_tensors[0].GetTensorMutableData<float>();
    auto gaze_yaws_data = this->output_tensors[1].GetTensorMutableData<float>();
    cv::Mat gaze_pitches_mat = cv::Mat(cv::Size(90, this->blob.size[0]), CV_32F);
    cv::Mat gaze_yaws_mat = cv::Mat(cv::Size(90, this->blob.size[0]), CV_32F);

    ::memcpy(gaze_pitches_mat.data, gaze_pitches_data, sizeof(float) * this->blob.size[0] * 90);
    ::memcpy(gaze_yaws_mat.data, gaze_yaws_data, sizeof(float) * this->blob.size[0] * 90);

    std::vector<float> idx_tensor_vec;
    for (size_t i = 0; i < 90; ++i) {
        idx_tensor_vec.emplace_back(static_cast<float>(i));
    }
    cv::Mat idx_tensor = cv::Mat(cv::Size(90, 1), CV_32F);
    ::memcpy(idx_tensor.data, idx_tensor_vec.data(), sizeof(float) * 90);

    for (size_t i = 0; i < this->blob.size[0]; ++i) {
        auto gaze_pitch_mat = gaze_pitches_mat(cv::Range((int) i, (int) i + 1), cv::Range::all());
        auto gaze_yaw_mat = gaze_yaws_mat(cv::Range((int) i, (int) i + 1), cv::Range::all());
        gaze_pitch_mat = softmax(gaze_pitch_mat);
        gaze_yaw_mat = softmax(gaze_yaw_mat);
        cv::Mat gaze_pitch_sum_mat, gaze_yaw_sum_mat;
        cv::multiply(gaze_pitch_mat, idx_tensor, gaze_pitch_sum_mat);
        cv::multiply(gaze_yaw_mat, idx_tensor, gaze_yaw_sum_mat);
        auto gaze_pitch_sum = cv::sum(gaze_pitch_sum_mat)[0];
        auto gaze_yaw_sum = cv::sum(gaze_yaw_sum_mat)[0];
        auto pitch_predicted = (gaze_pitch_sum * 4 - 180) * M_PI / 180;
        auto yaw_predicted = (gaze_yaw_sum * 4 - 180) * M_PI / 180;
        this->pitch_list.emplace_back(pitch_predicted);
        this->yaw_list.emplace_back(yaw_predicted);
    }
}

void GazeEstimate::inference() {
    std::vector<float> input_image_vec;
    input_image_vec.resize(this->blob.size[0] * 3 * 448 * 448);
    float *input_1 = input_image_vec.data();
    ::memcpy(input_1, this->blob.data, this->blob.size[0] * 3 * 448 * 448 * sizeof(float));

    std::vector<int64_t> input_shape_ = {this->blob.size[0], 3, 448, 448};

    auto input_tensor = Ort::Value::CreateTensor<float>(
            this->memory_info,
            input_1,
            input_image_vec.size(),
            input_shape_.data(),
            input_shape_.size()
    );

    this->output_tensors = this->session->Run(
            Ort::RunOptions{nullptr},
            this->input_node_names.data(),
            &input_tensor,
            1,
            this->output_node_names.data(),
            2
    );

}

#endif //GAZE_ESTIMATE_GAZE_ESTIMATE_HPP
