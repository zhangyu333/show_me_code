//
// Created by 张宇 on 2023/8/8.
//

#ifndef ARM_INVERSE_KINEMATICS___KINEMATICS_SOLVER_HPP
#define ARM_INVERSE_KINEMATICS___KINEMATICS_SOLVER_HPP

#include <cmath>
#include <vector>
#include <cassert>

namespace KinematicsSolver {
    // unit  cm  real Coordinate System  <<====>>   pixal Coordinate System
    static constexpr double P = 8;
    static constexpr double A1 = 20;
    static constexpr double A2 = 11;
    static constexpr double A3 = 11;
    static constexpr double A4 = 16;
    static constexpr double alpha = 180;

    inline double COS(double degrees) {
        return cos(degrees * M_PI / 180);
    }

    inline double SIN(double degrees) {
        return sin(degrees * M_PI / 180);
    }

    inline double ATAN2(double y, double x) {
        return atan2(y, x) * 180 / M_PI;
    }

    inline void cJ1(double &x, double &y, double &z, double &length, double &height, double &joint1) {
        height = z;
        length = sqrt(pow((y + P), 2) + pow(x, 2));
        if (length == 0) joint1 = 0;
        else joint1 = ATAN2((y + P), x) - 90;
    }

    inline void cJ3(double &L, double &H, double &joint3) {
        auto cos3 = (pow(L, 2) +
                     pow(H, 2) -
                     pow(A2, 2) -
                     pow(A3, 2)) / (2 * A2 * A3);
        assert(pow(cos3, 2) < 1);
        auto sin3 = sqrt(1 - pow(cos3, 2));
        joint3 = ATAN2(sin3, cos3);
    }

    inline void cJ2(double &L, double &H, double &joint3, double &joint2) {
        auto K1 = A2 + A3 * COS(joint3);
        auto K2 = A3 * SIN(joint3);
        auto w = ATAN2(K2, K1);
        joint2 = ATAN2(L, H) - w;
    }

    inline void cJ4(double &joint2, double &joint3, double &joint4) {
        joint4 = alpha - joint2 - joint3;
    }


    inline void inverseKinematics(double x, double y, double z, std::vector<double> &joints) {
        double length, height, joint1, joint2, joint3, joint4;
        cJ1(x, y, z, length, height, joint1);
        auto L = length - A4 * SIN(alpha);
        auto H = height - A4 * COS(alpha) - A1;
        cJ3(L, H, joint3);
        cJ2(L, H, joint3, joint2);
        cJ4(joint2, joint3, joint4);

        joint1 /= 90;

        joint2 /= 90;
        joint2 *= -1.5;

        joint3 /= 90;
        joint3 *= 1.25;

        joint4 /= 90;
        joint4 *= 1.5;

        if (joint1 == 0) joint1 = 0.1;
        joints = {joint1, joint2, joint3, joint4, 0, -1};
    }


}

#endif //ARM_INVERSE_KINEMATICS___KINEMATICS_SOLVER_HPP
