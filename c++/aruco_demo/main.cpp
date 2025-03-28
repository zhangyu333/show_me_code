#include <getopt.h>
#include <iostream>
#include <aruco/aruco.h>
#include <opencv2/opencv.hpp>

static struct option long_options[] = {
        {"image_path", required_argument, nullptr, 'i'},
        {"save_path",  required_argument, nullptr, 's'},
        {nullptr,      no_argument,       nullptr, 0}
};

int main(int argc, char *argv[]) {
    cv::InputArray a(3);
    a.isMatVector();
    std::string image_path;
    std::string save_path;
    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc,
                              argv,
                              "",
                              long_options,
                              &option_index)) != -1) {
        if (opt == 'i') {
            image_path = optarg;
        } else if (opt == 's') {
            save_path = optarg;
        } else {
            std::cout << "unknown args!!!!" << std::endl;
        }
    }
    std::string check_info = image_path.empty() ? "参数解析错误" : "参数解析成功";
    std::cout << "check args: " << check_info << std::endl;
    assert(!image_path.empty());
    auto image = cv::imread(image_path);
    aruco::MarkerDetector markerDetector;
    aruco::CameraParameters camera_param;
    markerDetector.setDetectionMode(aruco::DM_FAST, 0);
    clock_t st = clock();
    auto markers = markerDetector.detect(image);
    std::cout << "AR detect cost time: " << double(clock() - st) / CLOCKS_PER_SEC << "ms" << std::endl;
    aruco::MarkerPoseTracker markerPoseTracker;
    for (auto &mark: markers) {
        mark.draw(image);
        std::cout << mark.id << "=";
        for (int i = 0; i < 4; i++)
            std::cout << "(" << mark[i].x << "," << mark[i].y << ") ";
        std::cout << mark.getCenter();
        std::cout << "\n";
    }
    cv::imwrite(save_path, image);
    return 0;
}
