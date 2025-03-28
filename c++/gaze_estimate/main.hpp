//
// Created by 张宇 on 2024/6/27.
//
#pragma once

#ifndef GAZE_ESTIMATE_MAIN_HPP
#define GAZE_ESTIMATE_MAIN_HPP

#include "face_detect.hpp"
#include "gaze_estimate.hpp"

namespace GEM {
    __attribute__((unused)) FaceDetect fd_model(
            "/Users/zhangyu/Desktop/ml/_project_pytorch/人脸/models/fast_face_detect.onnx");
    __attribute__((unused)) GazeEstimate ge_model(
            "/Users/zhangyu/Desktop/ml/_project_pytorch/gaze_estimate/models/gaze_estimate.onnx");
    std::vector<cv::Mat> faces;

    __attribute__((unused)) void render(cv::Mat &image, PYRES &pyres, std::vector<std::vector<float>> &dets) {
        auto yaw_list = pyres.yaw_list;
        auto pitch_list = pyres.pitch_list;
        for (size_t i = 0; i < dets.size(); ++i) {
            auto yaw = yaw_list[i];
            auto pitch = pitch_list[i];

            auto bbox = dets[i];
            auto x1 = bbox[0];
            auto y1 = bbox[1];
            auto x2 = bbox[2];
            auto y2 = bbox[3];
            auto w = x2 - x1;
            auto h = y2 - y1;
            auto rect = cv::Rect(int(x1), int(y1), int(w), int(h));
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 1);

            auto pos1 = cv::Point(x1 + w / 2, y1 + h / 2);
            auto dx = -w * std::sin(pitch) * std::cos(yaw);
            auto dy = -w * std::sin(yaw);
            auto pos2 = cv::Point(pos1.x + dx, pos1.y + dy);
            cv::arrowedLine(image, pos1, pos2, cv::Scalar(255, 255, 0), 2, cv::LINE_AA, 0, 0.18);
        }

        cv::imshow("image", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }


    void call(const std::string &image_path) {
        faces.clear();
        auto image = cv::imread(image_path);
        auto detections = fd_model.call(image);
        for (auto &detection: detections) {
            auto x1 = detection[0];
            auto y1 = detection[1];
            auto x2 = detection[2];
            auto y2 = detection[3];
            auto w = x2 - x1;
            auto h = y2 - y1;
            auto rect = cv::Rect(int(x1), int(y1), int(w), int(h));
            auto face = image(rect);
            faces.emplace_back(face);
        }
        auto py_results = ge_model.call(faces);
        render(image, py_results, detections);
    }

}

#endif //GAZE_ESTIMATE_MAIN_HPP
