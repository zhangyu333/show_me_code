//
// Created by 张宇 on 2024/4/15.
//

#include "multi_view_3D_reconstruction.h"

void MultiViewStructure::SetViews(const std::string &image_root_path) {
    this->dcmk = mm.call(image_root_path);
}

void MultiViewStructure::SetK(double &fx, double &fy, double &x0, double &y0) {
    this->K = (cv::Mat_<float>(3, 3) <<
                                     fx, 0, x0,
            0, fy, y0,
            0, 0, 1);
}

ISI MultiViewStructure::initStructure() {
    printf("两视图重构初始化三维姿态 ==> ");
    auto match_point1 = MultiViewStructure::getMatchedPoints(this->dcmk.keypoints_all[0], this->dcmk.keypoints_all[1],
                                                             this->dcmk.matches_all[0]);
    auto match_color1 = MultiViewStructure::getMatchedColors(this->dcmk.colors_all[0], this->dcmk.colors_all[1],
                                                             this->dcmk.matches_all[0]);
    auto rtm = this->findTransform(match_point1.dst_pts, match_point1.src_pts);

    auto p1 = this->maskOutPoints<cv::Point2f>(match_point1.src_pts, rtm.mask);
    auto p2 = this->maskOutPoints<cv::Point2f>(match_point1.dst_pts, rtm.mask);
    auto c1 = this->maskOutPoints<cv::Vec3b>(match_color1.src_colors, rtm.mask);

    cv::Mat first_transform_matrix = (
            cv::Mat_<float>(3, 4) <<
                                  1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0
    );
    cv::Mat second_transform_matrix;
    cv::hconcat(rtm.rotation, rtm.translation, second_transform_matrix);
    second_transform_matrix.convertTo(second_transform_matrix, CV_32F);
    auto structure_ = this->reStruct(first_transform_matrix, second_transform_matrix, p1, p2);
    printf("   done\n");

    std::vector <std::vector<int>> correspond_struct_idx;
    for (size_t i = 0; i < this->dcmk.keypoints_all.size(); ++i) {
        std::vector<int> idxes;
        for (size_t j = 0; j < this->num_features; ++j) {
            idxes.emplace_back(-1);
        }
        correspond_struct_idx.emplace_back(idxes);
    }
    int idx = 0;
    for (size_t i = 0; i < this->dcmk.matches_all[0].size(); ++i) {
        auto match = this->dcmk.matches_all[0][i];
        if ((int) rtm.mask.at<char>(i) == 0) continue;
        correspond_struct_idx[0][(int) match.queryIdx] = idx;
        correspond_struct_idx[1][(int) match.trainIdx] = idx;
        idx += 1;
    }

    return ISI{structure_, correspond_struct_idx, c1, second_transform_matrix};
}

MatchPoints MultiViewStructure::getMatchedPoints(std::vector <cv::KeyPoint> &p1, std::vector <cv::KeyPoint> &p2,
                                                 std::vector <cv::DMatch> &matches) {

    std::vector <cv::Point2f> src_pts;
    std::vector <cv::Point2f> dst_pts;
    for (auto &match: matches) {
        auto src_pt = p1[match.queryIdx].pt;
        auto dst_pt = p2[match.trainIdx].pt;
        src_pts.emplace_back(src_pt);
        dst_pts.emplace_back(dst_pt);
    }

    return MatchPoints{src_pts, dst_pts};
}

MatchColors MultiViewStructure::getMatchedColors(std::vector <cv::Vec3b> &c1, std::vector <cv::Vec3b> &c2,
                                                 std::vector <cv::DMatch> &matches) {
    std::vector <cv::Vec3b> src_colors;
    std::vector <cv::Vec3b> dst_colors;
    for (auto &match: matches) {
        auto src_pt = c1[match.queryIdx];
        auto dst_pt = c2[match.trainIdx];
        src_colors.emplace_back(src_pt);
        dst_colors.emplace_back(dst_pt);
    }
    return MatchColors{src_colors, dst_colors};
}

RTM MultiViewStructure::findTransform(std::vector <cv::Point2f> &points1, std::vector <cv::Point2f> &points2) {
    cv::Mat mask, rotation, translation;
    auto focal_length = 0.5 * (this->K.at<float>(0, 0) + this->K.at<float>(1, 1));
    cv::Point2d pp = cv::Point2d(K.at<float>(0, 2), K.at<float>(1, 2));
    auto essentialMat = cv::findEssentialMat(points1, points2, focal_length, pp, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(essentialMat.t(), points1, points2, this->K, rotation, translation, mask);
    return RTM{mask, rotation, translation};
}

template<class T>
std::vector <T> MultiViewStructure::maskOutPoints(std::vector <T> objects, cv::Mat &mask) {
    std::vector <T> objects_copy;
    for (size_t i = 0; i < mask.size[0]; ++i) {
        if ((int) mask.at<char>(i) > 0) {
            objects_copy.emplace_back(objects[i]);
        }
    }
    return objects_copy;
}

cv::Mat MultiViewStructure::reStruct(cv::Mat &transform_matrix1, cv::Mat &transform_matrix2,
                                     std::vector <cv::Point2f> &p1, std::vector <cv::Point2f> &p2) {
    cv::Mat M1 = this->K * transform_matrix1;
    cv::Mat M2 = this->K * transform_matrix2;

    cv::Mat projPoints1(p1);
    cv::Mat projPoints2(p2);

    cv::Mat points4D;
    cv::triangulatePoints(
            M1,
            M2,
            projPoints1.t(),
            projPoints2.t(),
            points4D);

    cv::Mat points3D;
    cv::convertPointsFromHomogeneous(points4D.t(), points3D);
    return points3D;
}

cv::Mat MultiViewStructure::solvePnpGetRT(O3d2d &o3d2d) {
    cv::Mat R(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F); // 无畸变相机
    cv::Mat r = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat T = cv::Mat::zeros(3, 1, CV_64F);
    cv::solvePnPRansac(o3d2d.object_points, o3d2d.image_points, this->K, distCoeffs, r, T);
    cv::Rodrigues(r, R);
    cv::Mat transform_matrix;
    cv::hconcat(R, T, transform_matrix);
    transform_matrix.convertTo(transform_matrix, CV_32F);
    return transform_matrix;
}

O3d2d MultiViewStructure::getObjPointsImagePoints(
        std::vector <cv::DMatch> &matches,
        std::vector<int> &struct_indices,
        cv::Mat &structure,
        std::vector <cv::KeyPoint> &key_points
) {
    std::vector <cv::Point3f> object_points;
    std::vector <cv::Point2f> image_points;
    for (auto &match: matches) {
        auto query_idx = match.queryIdx;
        auto train_idx = match.trainIdx;
        auto struct_idx = struct_indices[query_idx];
        if (struct_idx < 0) continue;
        cv::Mat object_point = structure(cv::Range(int(struct_idx), int(struct_idx) + 1), cv::Range::all());
        object_points.emplace_back(
                cv::Point3f(object_point.at<float>(0), object_point.at<float>(1), object_point.at<float>(2)));
        auto image_point = key_points[train_idx].pt;
        image_points.emplace_back(cv::Point2f(image_point.x, image_point.y));
    }
    return O3d2d{
            object_points,
            image_points
    };
}

void MultiViewStructure::fusionStructure(std::vector <cv::DMatch> &matches, std::vector<int> &struct_indices,
                                         std::vector<int> &next_struct_indices, cv::Mat &structure,
                                         cv::Mat &next_structure, std::vector <cv::Vec3b> &colors,
                                         std::vector <cv::Vec3b> next_colors) {

    for (size_t i = 0; i < matches.size(); ++i) {
        auto match = matches[i];
        auto query_idx = match.queryIdx;
        auto train_idx = match.trainIdx;
        auto struct_idx = struct_indices[query_idx];
        if (struct_idx >= 0) {
            next_struct_indices[train_idx] = struct_idx;
            continue;
        }
        cv::Mat object_point = next_structure(cv::Range(i, i + 1), cv::Range::all());
        cv::vconcat(structure, object_point, structure);
        auto color = next_colors[i];
        colors.emplace_back(color);
        struct_indices[query_idx] = structure.size[0] - 1;
        next_struct_indices[train_idx] = structure.size[0] - 1;
    }

}

void MultiViewStructure::run() {
    auto isi = this->initStructure();
    auto transform_matrix = isi.second_transform_matrix;
    auto correspond_struct_idx = isi.correspond_struct_idx;
    this->structure = isi.structure;
    this->colors = isi.colors;

    printf("多视图重建中");
    for (size_t i = 1; i < this->dcmk.matches_all.size(); ++i) {
        auto matches = this->dcmk.matches_all[i];

        auto o3d2d = MultiViewStructure::getObjPointsImagePoints(
                matches,
                correspond_struct_idx[i],
                this->structure,
                this->dcmk.keypoints_all[i + 1]
        );
        auto next_transform_matrix = solvePnpGetRT(o3d2d);
        printf("[%zu/%lu] ", i, this->dcmk.matches_all.size() - 1);

        auto match_point = MultiViewStructure::getMatchedPoints(this->dcmk.keypoints_all[i],
                                                                this->dcmk.keypoints_all[i + 1],
                                                                this->dcmk.matches_all[i]);
        auto match_color = MultiViewStructure::getMatchedColors(this->dcmk.colors_all[i], this->dcmk.colors_all[i + 1],
                                                                this->dcmk.matches_all[i]);

        auto next_structure = reStruct(
                transform_matrix,
                next_transform_matrix,
                match_point.src_pts,
                match_point.dst_pts);
        transform_matrix = next_transform_matrix;


        MultiViewStructure::fusionStructure(
                matches,
                correspond_struct_idx[i],
                correspond_struct_idx[i + 1],
                this->structure,
                next_structure,
                this->colors,
                match_color.src_colors
        );

    }
    printf("多视图重建结束\n");

}

void MultiViewStructure::vis() {
    // 可视化
    auto length = this->structure.size[0];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud <pcl::PointXYZRGB>);
    cloud->width = length;
    cloud->height = 1;
    cloud->resize(length);

    std::vector<float> total_depth;

    for (size_t i = 0; i < length; ++i) {
        auto point = this->structure.at<cv::Point3f>(i);
        total_depth.emplace_back(point.z);
    }
    auto max_depth = std::max_element(total_depth.begin(), total_depth.end());
    int sum_depth = std::accumulate(total_depth.begin(), total_depth.end(), 0);
    double avg_depth = static_cast<double>(sum_depth) / total_depth.size();
    std::cout << "最大深度: " << *max_depth << "平均深度: " << avg_depth << std::endl;

    for (size_t i = 0; i < length; ++i) {
        auto point = this->structure.at<cv::Point3f>(i);
        auto color = this->colors[i];
        if (std::abs(avg_depth - point.z) > (avg_depth - 1)) continue;
        cloud->points[i].r = (int) color[2];
        cloud->points[i].g = (int) color[1];
        cloud->points[i].b = (int) color[0];
        cloud->points[i].x = point.x;
        cloud->points[i].y = point.y;
        cloud->points[i].z = point.z;
    }

//    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
//    sor.setInputCloud (cloud);
//    sor.setMeanK (100);  // 设置用于计算平均距离的邻近点数量
//    sor.setStddevMulThresh (1.0);  // 设置标准差的倍数作为阈值
//    sor.filter (*cloud_filtered);


    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField <pcl::PointXYZRGB> rgb(cloud);
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "SFM-DEMO");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "SFM-DEMO");
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}