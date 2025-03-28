//
// Created by 张宇 on 2024/4/15.
//

#ifndef SFM_DEMO_MY_MATCHER_H
#define SFM_DEMO_MY_MATCHER_H

#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct {
    std::vector<cv::Mat> descriptor_all;
    std::vector<std::vector<cv::Vec3b>> colors_all;
    std::vector<std::vector<cv::DMatch>> matches_all;
    std::vector<std::vector<cv::KeyPoint>> keypoints_all;

} DCMK;


class MyMatcher {
public:
    MyMatcher() = default;

    ~MyMatcher() = default;

    void matchFeatures();

    void extractFeatures(const std::string &image_root_path);

    std::vector<cv::DMatch> matchFeatures_(cv::Mat &query, cv::Mat &train);

    DCMK call(const std::string &image_root_path);

private:
    float MRT = 0.5;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(50000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<cv::Mat> descriptor_all;
    std::vector<std::vector<cv::Vec3b>> colors_all;
    std::vector<std::vector<cv::DMatch>> matches_all;
    std::vector<std::vector<cv::KeyPoint>> keypoints_all;
};


#endif //SFM_DEMO_MY_MATCHER_H
