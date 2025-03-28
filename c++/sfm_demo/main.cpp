#include "multi_view_3D_reconstruction.h"

int main() {
    MultiViewStructure mvs;
    std::string image_root_path = "/Users/zhangyu/CLionProjects/sfm_demo/ImageDataset_SceauxCastle/images";
    double fx = 2905.88, fy = 2905.88, x0 = 1416, y0 = 1064;
    mvs.SetViews(image_root_path);
    mvs.SetK(fx, fy, x0, y0);
    mvs.run();
    mvs.vis();

    return 0;
}