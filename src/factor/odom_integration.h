#pragma once

#include <eigen3/Eigen/Dense>
#include "../parameters.h"
#include "../utility.h"
using namespace Eigen;

class OdomIntegration
{
public:
    OdomIntegration()
        : delta_odom(Vector3d::Zero()), covariance(Matrix3d::Zero()) // 设置初始值，并用来处理odom积分
    {
        odom_noise.setZero();
        odom_noise(0, 0) = ODOM_X_N * ODOM_X_N;
        odom_noise(1, 1) = ODOM_Y_N * ODOM_Y_N;
        odom_noise(2, 2) = ODOM_TH_N * ODOM_TH_N;
    }

    void push_back(const Vector3d &delta_odom_k)
    {
        Matrix2d phi_ik = Rotation2Dd(delta_odom(2)).toRotationMatrix(); // 设置Rotation
        delta_odom.head(2) += phi_ik * delta_odom_k.head(2);             // 设置odom朝向1
        delta_odom(2) += delta_odom_k(2);                                // 将odom加入
        delta_odom(2) = Utility::normalizeAngle(delta_odom(2));

        Matrix3d Ak, Bk;
        Ak.setIdentity();
        Bk.setIdentity();
        Ak.block<2, 1>(0, 2) = phi_ik * Utility::skewSymmetric(1.0) * delta_odom_k.head(2); // 误差传递
        Bk.block<2, 2>(0, 0) = phi_ik;
        covariance = Ak * covariance * Ak.transpose() + Bk * odom_noise * Bk.transpose(); // 计算cov
    }

    Vector3d delta_odom;
    Matrix3d covariance;
    Matrix3d odom_noise;
};