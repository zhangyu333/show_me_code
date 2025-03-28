//
// Created by 张宇 on 2024/4/15.
//

#include "my_matcher.h"

void MyMatcher::extractFeatures(const std::string &image_root_path) {
    this->keypoints_all.clear();
    this->descriptor_all.clear();
    cv::Mat descriptors, gray;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<std::string> image_paths;
    std::vector<cv::Vec3b> colors;
    cv::glob(image_root_path + "/*", image_paths);
    printf("特征提取中");
    for (size_t i = 0; i < image_paths.size(); ++i) {
        auto image_path = image_paths[i];
        auto bgr_image = cv::imread(image_path);
        colors.clear();
        cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
        MyMatcher::detector->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);
        printf("[%zu/%lu] ", i + 1, image_paths.size());
        if (keypoints.size() < 10) continue;
        this->keypoints_all.emplace_back(keypoints);
        this->descriptor_all.emplace_back(descriptors);
        for (auto &keypoint: keypoints) {
            auto p = keypoint.pt;
            cv::Vec3b bgr = bgr_image.at<cv::Vec3b>((int) p.y, (int) p.x);
            colors.emplace_back(bgr);
        }
        colors_all.emplace_back(colors);
    }
    printf("特征提取结束\n");
}


void MyMatcher::matchFeatures() {
    printf("ORB特征匹配中");
    this->matches_all.clear();
    for (size_t i = 1; i < this->descriptor_all.size(); ++i) {
        cv::Mat query = this->descriptor_all[i - 1];
        cv::Mat train = this->descriptor_all[i];
        auto matches = this->matchFeatures_(query, train);
        printf("[%zu/%lu] ", i, this->descriptor_all.size() - 1);
        this->matches_all.emplace_back(matches);
    }
    printf("ORB特征匹配结束\n");
}

std::vector<cv::DMatch> MyMatcher::matchFeatures_(cv::Mat &query, cv::Mat &train) {
    std::vector<cv::DMatch> good_matches;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    this->matcher->knnMatch(query, train, knn_matches, 2);
    for (auto &knn_matche: knn_matches) {
        if (knn_matche[0].distance < this->MRT * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    return good_matches;
}

DCMK MyMatcher::call(const std::string &image_root_path) {
    this->extractFeatures(image_root_path);
    this->matchFeatures();
    return DCMK{this->descriptor_all, this->colors_all, this->matches_all, this->keypoints_all};
}
