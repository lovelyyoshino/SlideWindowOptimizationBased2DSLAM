#include "feature_manager.h"

int FeaturePerId::endFrame() // 返回该特征点最后一次被观测到的帧号
{
    return start_frame + feature_per_frame.size() - 1;
}

int FeatureManager::getFeatureCount()
{
    return features.size();
}
MatrixXd FeatureManager::getFeaturePose() // 将特征点的世界坐标转为矩阵形式
{
    int f_cnt = features.size();
    MatrixXd f_pose(f_cnt, 2);
    int feature_index = -1;
    for (auto &it_per_id : features)
    {
        f_pose.row(++feature_index) = it_per_id.pose.transpose();
    }
    return f_pose;
}

void FeatureManager::setFeaturePose(const MatrixXd &f_pose) // 将特征点的世界坐标赋值给特征点
{
    int feature_index = -1;
    for (auto &it_per_id : features)
    {
        it_per_id.pose = f_pose.row(++feature_index).transpose();
    }
}

void FeatureManager::addFeature(int frame_count, const vector<pair<int, Vector2d>> obs) // 添加特征点
{
    printf("[addFeature]input feature: %d\n", static_cast<int>(obs.size()));
    printf("[addFeature]num of feature: %d\n", getFeatureCount());

    new_feature_num = 0;
    last_track_num = 0;

    for (auto &id_point : obs)
    {
        FeaturePerFrame f_per_fra(id_point.second); // 特征点在当前帧的观测
        int feature_id = id_point.first;            // 特征点id
        auto it = find_if(features.begin(), features.end(), [feature_id](const FeaturePerId &it)
                          { return it.feature_id == feature_id; }); // 查找是否已经存在该特征点
        // new feature
        if (it == features.end())
        {
            features.push_back(FeaturePerId(feature_id, frame_count)); // 特征点第一次被观测到的帧
            features.back().feature_per_frame.push_back(f_per_fra);    // 特征点在当前帧的观测
            new_feature_num++;
        }
        // old feature
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra); // 特征点在当前帧的观测
            last_track_num++;
        }
    }
}

void FeatureManager::initializeNewFeaturePose(int frame_count, const Matrix2d &R, const Vector2d &P) // 初始化新的特征点的世界坐标
{
    for (auto &it : features)
    {
        if (it.start_frame == frame_count)
        {
            Vector2d Prf;
            double r = it.feature_per_frame[0].range; // 特征点在当前帧的观测
            double phi = it.feature_per_frame[0].theta;
            Prf << r * cos(phi), r * sin(phi);
            it.pose = R * Prf + P;
        }
    }
}
