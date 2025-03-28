//
// Created by 张宇 on 2024/6/27.
//

#ifndef GAZE_ESTIMATE_FACE_DETECT_HPP
#define GAZE_ESTIMATE_FACE_DETECT_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

typedef struct {
    cv::Mat org_image;
    cv::Mat input_image;
    int w;
    int h;
} FD_PRE_RESULT;

class FaceDetect {

public:
    explicit FaceDetect(const std::string &model_path);

    ~FaceDetect() = default;

    std::vector<std::vector<float>> call(cv::Mat &input_image);

private:

    void inference();

    void imPreProc(cv::Mat &frame);

    void postProc();

    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"scores", "boxes"};
    std::vector<int64_t> input_shape_{1, 3, 240, 320};
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "FD");
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);
    __attribute__((unused)) std::vector<Ort::Value> output_tensors;

    FD_PRE_RESULT fd_pre_result;
    cv::Mat scores = cv::Mat(cv::Size(2, 4420), CV_32F);
    cv::Mat boxes = cv::Mat(cv::Size(4, 4420), CV_32F);

    float prob_threshold = 0.7;
    float iou_threshold = 0.3;

    std::vector<std::vector<float>> post_results;
};

FaceDetect::FaceDetect(const std::string &model_path) {
    this->session_options.SetIntraOpNumThreads(1);
    this->session_options.SetInterOpNumThreads(1);
    this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
}

std::vector<std::vector<float>> FaceDetect::call(cv::Mat &input_image) {
    clock_t st = clock();
    this->imPreProc(input_image);
    this->inference();
    this->postProc();
    clock_t et = clock();
    printf("人脸检测推理耗时: %fms\n", ((double) (et - st) / CLOCKS_PER_SEC) * 1000);
    return this->post_results;
}

void FaceDetect::imPreProc(cv::Mat &frame) {
    cv::Mat resize_image;
    auto h = frame.rows;
    auto w = frame.cols;
    cv::resize(frame, resize_image, cv::Size(320, 240));
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);
    resize_image.convertTo(resize_image, CV_32FC3);
    resize_image -= cv::Scalar(127.0, 127.0, 127.0);
    resize_image /= cv::Scalar(128.0, 128.0, 128.0);
    auto inp = cv::dnn::blobFromImage(resize_image);
    fd_pre_result.org_image = frame;
    fd_pre_result.input_image = inp;
    fd_pre_result.w = w;
    fd_pre_result.h = h;
}


void FaceDetect::inference() {
    std::vector<float> input_image;
    input_image.resize(3 * 320 * 240);
    float *input_1 = input_image.data();
    ::memcpy(input_1,
             this->fd_pre_result.input_image.data,
             3 * 240 * 320 * sizeof(float));

    auto input_tensor = Ort::Value::CreateTensor<float>(
            this->memory_info,
            input_1,
            input_image.size(),
            this->input_shape_.data(),
            this->input_shape_.size());

    this->output_tensors = this->session->Run(
            Ort::RunOptions{nullptr},
            this->input_node_names.data(),
            &input_tensor,
            this->input_node_names.size(),
            this->output_node_names.data(),
            this->output_node_names.size()
    );
}

inline float iou(const std::vector<float> &box1, const std::vector<float> &box2) {
    float area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    auto inner_x1 = std::max(box1[0], box2[0]);
    auto inner_y1 = std::max(box1[1], box2[1]);
    auto inner_x2 = std::min(box1[2], box2[2]);
    auto inner_y2 = std::min(box1[3], box2[3]);
    auto inner_w = (inner_x2 - inner_x1) > 0 ? inner_x2 - inner_x1 : 0;
    auto inner_h = (inner_y2 - inner_y1) > 0 ? inner_y2 - inner_y1 : 0;
    auto inner_area = inner_h * inner_w;
    return inner_area / (area_box1 + area_box2 - inner_area);
}

void FaceDetect::postProc() {
    auto score_fdata = this->output_tensors[0].GetTensorMutableData<float>();
    auto boxes_fdata = this->output_tensors[1].GetTensorMutableData<float>();
    ::memcpy(this->scores.data, score_fdata, sizeof(float) * 4420 * 2);
    ::memcpy(this->boxes.data, boxes_fdata, sizeof(float) * 4420 * 4);

    this->post_results.clear();
    for (int i = 0; i < 4420; ++i) {
        if (i == 0) continue;
        auto score = this->scores({i - 1, i}, {1, 2});
        auto box = this->boxes({i - 1, i}, {0, 4});
        auto prob = score.at<float>(0);
        if (prob < this->prob_threshold) continue;
        auto x1 = box.at<float>(0);
        auto y1 = box.at<float>(1);
        auto x2 = box.at<float>(2);
        auto y2 = box.at<float>(3);
        this->post_results.push_back({x1, y1, x2, y2, prob});
    }
    std::sort(this->post_results.begin(), this->post_results.end(),
              [](const std::vector<float> &box1, const std::vector<float> &box2) -> bool { return box1[4] > box2[4]; });
    std::vector<std::vector<float>> nms_results;
    while (!this->post_results.empty()) {
        auto res = this->post_results[0];
        nms_results.emplace_back(res);
        this->post_results.erase(this->post_results.begin());
        std::vector<int> erase_idxes;
        auto it = this->post_results.begin();
        while (it != this->post_results.end()) {
            float res_iou = iou(res, (*it));
            if (res_iou > this->iou_threshold) {
                this->post_results.erase(it);
            } else {
                it++;
            }
        }
    }
    for (auto &face: nms_results) {
        face[0] *= (float) this->fd_pre_result.w;
        face[1] *= (float) this->fd_pre_result.h;
        face[2] *= (float) this->fd_pre_result.w;
        face[3] *= (float) this->fd_pre_result.h;
    }
    this->post_results = nms_results;
}


#endif //GAZE_ESTIMATE_FACE_DETECT_HPP
