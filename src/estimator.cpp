#include "estimator.h"
#include "visualization.h"

Estimator::Estimator()
{
    printf("system begins.\n");
    clearState();
    Vector2d initP = Vector2d::Zero();  // 设置初始位置为0
    Matrix2d initR = Utility::toR(0.0); // 设置初始旋转矩阵为单位矩阵
    initFirstPose(initP, initR);
    // 主要线程，处理测量，估计状态
    processThread = std::thread(&Estimator::processMeasurements, this); // 创建线程
}

Estimator::~Estimator()
{
}
void Estimator::clearState() // 清空状态
{
    while (!odomBuf.empty())// 清空里程计缓存
        odomBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();
    preTime = -1;
    curTime = 0;
    latest_P.setZero();
    latest_R.setIdentity();
    for (int i = 0; i < WINDOW_SIZE + 1; ++i)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        if (pre_odom_integrations[i] != nullptr)
            delete pre_odom_integrations[i];
        pre_odom_integrations[i] = nullptr;
    }
    frame_count = 0;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;
    last_marginalization_info = nullptr;
}

void Estimator::initFirstPose(Eigen::Vector2d p, Eigen::Matrix2d r)
{
    Ps[0] = p;
    Rs[0] = r;
}

// 线程1： feature回调函数线程
void Estimator::inputFeature(double t, const std::vector<std::pair<int, Eigen::Vector2d>> &features)
{
    mBuf.lock();                             // 锁住缓冲区
    featureBuf.push(make_pair(t, features)); // 将特征点放入缓冲区
    mBuf.unlock();
}

// 线程2： odom回调函数线程
void Estimator::inputOdom(double t, const Eigen::Vector3d &d_odom_meas)
{
    mBuf.lock();
    odomBuf.push(make_pair(t, d_odom_meas));
    mBuf.unlock();

    // Fast predict robot pose according to odom measurement
    mPropagate.lock();            // 锁住预测
    fastPredictOdom(d_odom_meas); // 预测
    mPropagate.unlock();

    pubLatestOdometry(latest_P, latest_R, t);
}

void Estimator::fastPredictOdom(const Eigen::Vector3d &delta_odom)
{
    latest_P += latest_R * delta_odom.head(2); // 计算预测
    latest_R *= Utility::toR(delta_odom(2));
}

bool Estimator::odomAvailable(double t)
{
    if (!odomBuf.empty() && t <= odomBuf.back().first) // 判断odom是否可用
        return true;
    else
        return false;
}

bool Estimator::getOdomInterval(double t0, double t1, vector<pair<double, Vector3d>> &odomVector) // 获取odom数据
{
    if (odomBuf.empty())
    {
        printf("not receive odom\n");
        return false;
    }
    if (t1 <= odomBuf.back().first)
    {
        while (odomBuf.front().first <= t0) // 将odom数据放入odomVector
        {
            odomBuf.pop();
        }
        while (odomBuf.front().first <= t1) // 将odom数据放入odomVector
        {
            odomVector.push_back(odomBuf.front()); // 将odom数据放入odomVector
            odomBuf.pop();
        }
    }
    else
    {
        printf("wait for odom\n");
        return false;
    }

    return true;
}

void Estimator::processOdom(double t, const Eigen::Vector3d &delta_odom) // 处理odom数据
{
    if (!pre_odom_integrations[frame_count])
        pre_odom_integrations[frame_count] = new OdomIntegration(); // ODOM 积分
    if (frame_count != 0)
    {
        pre_odom_integrations[frame_count]->push_back(delta_odom); // 预积分

        int j = frame_count;
        Ps[j] += Rs[j] * delta_odom.head(2); // 设置机器人的位置信息
        Rs[j] *= Utility::toR(delta_odom(2));
    }
}

void Estimator::processLandMarks(double t, const vector<pair<int, Vector2d>> &observations)
{
    printf("new landmarks coming -------------------\n");
    printf("Adding landmark numbers %lu \n", observations.size());
    ROS_DEBUG("[processLandMarks] frame_count %d", frame_count);

    time_0 = ros::Time::now();

    f_manager.addFeature(frame_count, observations);                                   // 添加特征点
    f_manager.initializeNewFeaturePose(frame_count, Rs[frame_count], Ps[frame_count]); // 初始化特征点的位置

    time_1 = ros::Time::now();
    elapsed_time = (time_1 - time_0).toSec();
    ROS_INFO("addFeature time: %f", elapsed_time);

    if (frame_count == WINDOW_SIZE) // 如果窗口满了
    {

        optimization(); // 优化

        key_poses.clear();                     // 清空关键帧
        for (int i = 0; i <= WINDOW_SIZE; i++) // 将窗口中的位姿作为关键帧
        {
            key_poses.push_back(Ps[i]); // 将窗口中的位姿作为关键帧
        }

        time_0 = ros::Time::now();

        slideWindow();        // 滑动窗口处理
        updateLatestStates(); // 更新最新的状态

        time_1 = ros::Time::now();
        elapsed_time = (time_1 - time_0).toSec();
        ROS_INFO("slideWindow time: %f", elapsed_time);
    }
    else
    {
        frame_count++;
        int prev_frame = frame_count - 1;
        Ps[frame_count] = Ps[prev_frame]; // 否则设置位姿
        Rs[frame_count] = Rs[prev_frame];
    }
}

void Estimator::updateLatestStates() // 更新最新的状态
{
    mPropagate.lock();
    latest_P = Ps[frame_count]; // 更新最新的位姿
    latest_R = Rs[frame_count];
    mBuf.lock();
    std::queue<pair<double, Vector3d>> tmp_odomBuf = odomBuf; // 传入buf
    mBuf.unlock();
    while (!tmp_odomBuf.empty())
    {
        Eigen::Vector3d delta_odom = tmp_odomBuf.front().second;
        fastPredictOdom(delta_odom); // 预测
        tmp_odomBuf.pop();           // 释放这个元素
    }
    mPropagate.unlock();
}

void Estimator::optimization()
{
    time_0 = ros::Time::now();
    vector2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = nullptr;
    // loss_function = new ceres::HuberLoss(1.0);
    // 添加robot pose参数块
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new Pose2DLocalParameterization();
        problem.AddParameterBlock(para_pose[i], 3, local_parameterization); // 添加robot pose参数块,并对每个pose进行优化
    }
    problem.SetParameterBlockConstant(para_pose[0]); // 设置第一帧的pose为常数

    // 添加边缘化信息
    if (last_marginalization_info)
    {
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info); // 添加边缘化信息
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);        // 添加边缘化信息
    }
    // 添加landmark观测约束
    int feature_index = -1;
    for (auto &it_per_id : f_manager.features)
    {
        ++feature_index;
        int frame_i = it_per_id.start_frame;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            Vector2d obs(it_per_frame.range, it_per_frame.theta);
            LandMarkFactor *landmark_factor = new LandMarkFactor(obs);                                                 // 添加landmark观测约束
            problem.AddResidualBlock(landmark_factor, loss_function, para_pose[frame_i], para_feature[feature_index]); // 添加landmark观测约束
            frame_i++;
        }
    }

    // 添加odom测量约束
    for (int i = 0; i < frame_count; i++)
    {
        int j = i + 1;
        OdomFactor *odom_factor = new OdomFactor(pre_odom_integrations[j]);
        problem.AddResidualBlock(odom_factor, loss_function, para_pose[i], para_pose[j]); // 添加odom测量约束
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = 0.04;

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary); // 优化
    time_1 = ros::Time::now();
    elapsed_time = (time_1 - time_0).toSec();
    ROS_INFO("solve time: %f", elapsed_time);

    printf("Iterations: %d\n", static_cast<int>(summary.iterations.size()));
    double2vector();
    // 1）将上一次先验残差项传递给marginalization_info

    // 2）将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中

    // 3）将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中

    // 4）计算每个残差，对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中

    // 5）多线程构造先验项舒尔补AX = b的结构，在X0处线性化计算Jacobian和残差

    // 6）调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
    MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // 添加边缘化信息
    vector2double();
    if (last_marginalization_info) // 如果有最近的边缘化信息
    {
        vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) // 将最近的边缘化信息中的pose参数块加入到drop_set中
        {
            if (last_marginalization_parameter_blocks[i] == para_pose[0]) // 将所有的param设置为起始位置
            {
                drop_set.push_back(i);
            }
        }
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);                                          // 添加边缘化信息
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set); // 添加边缘化信息
        marginalization_info->addResidualBlockInfo(residual_block_info);                                                                               // 设置边缘化信息
    }

    OdomFactor *odom_factor = new OdomFactor(pre_odom_integrations[1]); // 添加odom测量约束
    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(odom_factor, NULL, vector<double *>{para_pose[0], para_pose[1]}, vector<int>{0});
    marginalization_info->addResidualBlockInfo(residual_block_info); // 设置边缘化信息

    feature_index = -1;
    for (auto &it_per_id : f_manager.features) // 添加landmark观测约束
    {
        ++feature_index;
        int frame_k = it_per_id.start_frame;
        if (frame_k != 0)
            continue;
        for (auto &it_per_frame : it_per_id.feature_per_frame) // 需要对每个landmark进行优化
        {
            Eigen::Vector2d obs(it_per_frame.range, it_per_frame.theta);
            LandMarkFactor *landmark_factor = new LandMarkFactor(obs);
            ResidualBlockInfo *residual_block_info;
            if (frame_k == 0)
                residual_block_info = new ResidualBlockInfo(landmark_factor, NULL, vector<double *>{para_pose[0], para_feature[feature_index]}, vector<int>{0, 1});
            else
                residual_block_info = new ResidualBlockInfo(landmark_factor, NULL, vector<double *>{para_pose[frame_k], para_feature[feature_index]}, vector<int>{1});
            marginalization_info->addResidualBlockInfo(residual_block_info); // 设置边缘化信息
            ++frame_k;
        }
    }

    time_0 = ros::Time::now();

    marginalization_info->preMarginalize(); // 设置每个残差块的雅克比矩阵

    marginalization_info->marginalize(); // 边缘化

    time_1 = ros::Time::now();
    elapsed_time = (time_1 - time_0).toSec();
    ROS_INFO("marginalization time: %f", elapsed_time);

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++)
    {
        addr_shift[reinterpret_cast<long>(para_pose[i])] = para_pose[i - 1]; // 将para_pose[1]到para_pose[WINDOW_SIZE]的地址映射到para_pose[0]到para_pose[WINDOW_SIZE-1]的地址
    }
    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
    if (last_marginalization_info)
        delete last_marginalization_info;
    last_marginalization_info = marginalization_info;         // 将最近时刻的边缘化信息保存起来
    last_marginalization_parameter_blocks = parameter_blocks; // 将边缘化信息中的参数块保存起来
}

void Estimator::slideWindow()
{
    if (frame_count == WINDOW_SIZE)
    {
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            Rs[i].swap(Rs[i + 1]); // 交换两个矩阵
            Ps[i].swap(Ps[i + 1]);
            std::swap(pre_odom_integrations[i], pre_odom_integrations[i + 1]); // 交换两个指针
        }
        Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1]; // 将倒数第二个赋值给最后一个
        Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
        delete pre_odom_integrations[WINDOW_SIZE]; // 释放内存
        pre_odom_integrations[WINDOW_SIZE] = nullptr;
    }

    auto it = f_manager.features.begin();
    while (it != f_manager.features.end())
    {
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 删除第一个元素
            if (it->feature_per_frame.size() == 0)
            {
                it = f_manager.features.erase(it); // 删除it指向的元素，并返回下一个元素的迭代器
                continue;
            }
        }
        it++;
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_pose[i][0] = Ps[i](0); // Ps[i]是Vector2d类型，Ps[i](0)是double类型
        para_pose[i][1] = Ps[i](1);
        para_pose[i][2] = Utility::fromR(Rs[i]);
    }
    MatrixXd features_pose = f_manager.getFeaturePose();
    for (int i = 0; i < features_pose.rows(); i++)
    {
        para_feature[i][0] = features_pose.row(i)(0); // features_pose.row(i)(0)是double类型
        para_feature[i][1] = features_pose.row(i)(1);
    }
}

void Estimator::double2vector()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = Utility::toR(para_pose[i][2]);
        Ps[i] = Vector2d(para_pose[i][0], para_pose[i][1]);
    }
    int f_cnt = f_manager.getFeatureCount();
    MatrixXd f_pose(f_cnt, 2);
    for (int i = 0; i < f_cnt; i++)
    {
        f_pose.row(i)[0] = para_feature[i][0];
        f_pose.row(i)[1] = para_feature[i][1];
    }
    f_manager.setFeaturePose(f_pose);
}

// 线程3: processThread
void Estimator::processMeasurements()
{
    while (1)
    {
        pair<double, vector<pair<int, Vector2d>>> feature;
        vector<pair<double, Vector3d>> odomVector;
        if (!featureBuf.empty())
        {
            feature = featureBuf.front();
            curTime = feature.first;
            // 等待足够的odom测量序列
            while (1)
            {
                if (odomAvailable(curTime))
                    break;
                else
                {
                    printf("wait for odom ...\n");
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }

            // 获得上一帧和当前帧间的odom测量序列
            mBuf.lock();
            getOdomInterval(preTime, curTime, odomVector);
            ROS_DEBUG("get odom num: %d", (int)odomVector.size());
            featureBuf.pop();
            mBuf.unlock();

            time_0 = ros::Time::now();

            // 处理里程计测量值，进行预积分
            for (size_t i = 0; i < odomVector.size(); ++i)
            {
                processOdom(odomVector[i].first, odomVector[i].second);
            }

            time_1 = ros::Time::now();
            elapsed_time = (time_1 - time_0).toSec();
            ROS_INFO("processOdom time: %f", elapsed_time);

            // 处理特征点测量
            processLandMarks(feature.first, feature.second);

            preTime = curTime;

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);
            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubPointCloud(*this, header);
            pubTF(*this, header);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}
