//
// Created by 张宇 on 2024/4/15.
//

#ifndef SFM_DEMO_MULTI_VIEW_3D_RECONSTRUCTION_H
#define SFM_DEMO_MULTI_VIEW_3D_RECONSTRUCTION_H

#include "my_matcher.h"
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

typedef struct {
    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
} MatchPoints;

typedef struct {
    std::vector<cv::Vec3b> src_colors;
    std::vector<cv::Vec3b> dst_colors;
} MatchColors;

typedef struct {
    cv::Mat mask;
    cv::Mat rotation;
    cv::Mat translation;
} RTM;

typedef struct {
    cv::Mat structure;
    std::vector<std::vector<int>> correspond_struct_idx;
    std::vector<cv::Vec3b> colors;
    cv::Mat second_transform_matrix;
} ISI;

typedef struct {
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
} O3d2d;

class MultiViewStructure {
public:
    MultiViewStructure() = default;

    ~MultiViewStructure() = default;

    void run();

    ISI initStructure();

    static void fusionStructure(std::vector<cv::DMatch> &matches,
                                std::vector<int> &struct_indices,
                                std::vector<int> &next_struct_indices,
                                cv::Mat &structure,
                                cv::Mat &next_structure,
                                std::vector<cv::Vec3b> &colors,
                                std::vector<cv::Vec3b> next_colors
    );

    cv::Mat solvePnpGetRT(O3d2d &o3d2d);

    static O3d2d getObjPointsImagePoints(std::vector<cv::DMatch> &matches,
                                         std::vector<int> &struct_indices,
                                         cv::Mat &structure,
                                         std::vector<cv::KeyPoint> &key_points);

    void SetViews(const std::string &image_root_path);

    void SetK(double &fx, double &fy, double &x0, double &y0);

    template<class T>
    std::vector<T> maskOutPoints(std::vector<T> objects, cv::Mat &mask);

    cv::Mat reStruct(cv::Mat &transform_matrix1, cv::Mat &transform_matrix2, std::vector<cv::Point2f> &p1,
                     std::vector<cv::Point2f> &p2);

    static MatchPoints
    getMatchedPoints(std::vector<cv::KeyPoint> &p1, std::vector<cv::KeyPoint> &p2, std::vector<cv::DMatch> &matches);

    static MatchColors
    getMatchedColors(std::vector<cv::Vec3b> &c1, std::vector<cv::Vec3b> &c2, std::vector<cv::DMatch> &matches);

    RTM findTransform(std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2);

    void vis();
private:
    MyMatcher mm;
    DCMK dcmk;

    cv::Mat K;
    int num_features = 50000;

    cv::Mat structure;
    std::vector<cv::Vec3b> colors;
};


#endif //SFM_DEMO_MULTI_VIEW_3D_RECONSTRUCTION_H
