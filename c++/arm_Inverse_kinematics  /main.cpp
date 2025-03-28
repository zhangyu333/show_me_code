#include "kinematics_solver.hpp"
#include <iostream>


int main() {
    double x = 0;
    double y = 5;
    double z = 0;
    std::vector<double> joints;
    KinematicsSolver::inverseKinematics(x, y, z, joints);
    for (auto &item: joints) {
        std::cout << "     " << item;
    }
    return 0;
}
