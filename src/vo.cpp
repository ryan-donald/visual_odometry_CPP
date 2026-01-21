#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

int main() {

    std::cout << "input sequence to be utilized: ";

    std::string sequence;

    std::cin >> sequence;

    std::string left_directory_path = "../../../ryan_slam/kitti/sequences/" + sequence + "/image_0/*.png";
    std::string right_directory_path = "../../../ryan_slam/kitti/sequences/" + sequence + "/image_1/*.png";

    std::vector<cv::String> left_images;
    std::vector<cv::String> right_images;


    cv::glob(left_directory_path, left_images, false);
    cv::glob(right_directory_path, right_images, false);

    if (left_images.empty() || right_images.empty()) {
        std::cerr << "no images found" << "\n";
        return -1;
    }

    std::string camera_calib_path = "../../../ryan_slam/kitti/sequences/" + sequence + "/calib.txt";
    std::ifstream file(camera_calib_path);

    std::string line;

    float left_image_calibration[3][4];
    float right_image_calibration[3][4];

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string label;
        iss >> label;

        label = label.substr(0, label.length() - 1);

        float* target_matrix = nullptr;

        if (label == "P0") {
            target_matrix = &left_image_calibration[0][0];
        } else if (label == "P1") {
            target_matrix = &right_image_calibration[0][0];
        } else {
            continue;
        }

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> target_matrix[i])) {
                std::cerr << "Error reading value " << i << " from " << label << "\n";
                return false;
            }
        }
    }
    
    file.close();

    float cx = left_image_calibration[0][2];
    float cy = left_image_calibration[1][2];
    float fx = left_image_calibration[0][0];
    float fy = left_image_calibration[1][1];

    float baseline = std::abs(left_image_calibration[0][3] - right_image_calibration[0][3]) / left_image_calibration[0][0];\
    cv::Mat curr_pose_transform = cv::Mat::eye(4, 4, CV_64F); 

    int minDisparity = 0;
    int numDisparities = 128;
    int blockSize = 5;
    int P1 = 8 * 3 * blockSize * blockSize;
    int P2 = 32 * 3 * blockSize * blockSize;
    int disp12MaxDiff = 1;
    int preFilterCap = 63;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 32;

    cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        disp12MaxDiff,
        preFilterCap,
        uniquenessRatio,
        speckleWindowSize,
        speckleRange,
        cv::StereoSGBM::MODE_SGBM_3WAY
    );

    int nFeatures = 6000;
    float scaleFactor = 1.2;
    int nLevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    int fastThreshold = 15;

    cv::Ptr<cv::ORB> detector = cv::ORB::create(
        nFeatures,
        scaleFactor,
        nLevels,
        edgeThreshold,
        firstLevel,
        WTA_K,
        cv::ORB::HARRIS_SCORE,
        patchSize,
        fastThreshold
    );

    cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
    );

    std::ofstream trajectory_file("trajectory_" + sequence + ".txt");
    if (!trajectory_file.is_open()) {
        std::cerr << "Failed to open trajectory.txt for writing" << "\n";
        return -1;
    }

    trajectory_file << "1 0 0 0 0 1 0 0 0 0 1 0" << "\n";


    std::cout << "starting visual odometry calculation for sequence " 
              << sequence << " with " << left_images.size() - 1 << " frame pairs" << "\n";

    for (size_t i = 0; i < left_images.size() - 1; i++) {

        cv::Mat left_image = cv::imread(left_images[i]);
        cv::Mat right_image = cv::imread(right_images[i]);
        cv::Mat next_left_image = cv::imread(left_images[i+1]);

        if (left_image.empty() || right_image.empty() || next_left_image.empty()) {
            std::cerr << "Failed to load images at frame " << i << "\n";
            return -1;
        }

        cv::Mat left_gray, right_gray, next_left_gray;
        cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(next_left_image, next_left_gray, cv::COLOR_BGR2GRAY);

        cv::Mat disp_map;

        matcher->compute(left_gray, right_gray, disp_map);

        cv::Mat disp_float;
        disp_map.convertTo(disp_float, CV_32F, 1.0/16.0);

        cv::Mat depth_map = cv::Mat::zeros(disp_float.size(), CV_64F);

        for (int row = 0; row < disp_float.rows; ++row) {
            for (int col = 0; col < disp_float.cols; ++col) {
                float disp = disp_float.at<float>(row, col);
                depth_map.at<double>(row, col) = (fx * baseline) / (disp + 1e-6);  // Match Python's approach
            }
        } 

        std::vector<cv::KeyPoint> left_keypoints;
        std::vector<cv::KeyPoint> left_next_keypoints;
        cv::Mat left_descriptors;
        cv::Mat left_next_descriptors;

        detector->detectAndCompute(left_gray, cv::noArray(), left_keypoints, left_descriptors);
        detector->detectAndCompute(next_left_gray, cv::noArray(), left_next_keypoints, left_next_descriptors);

        std::vector<std::vector<cv::DMatch>> knn_matches;
        
        if ((left_descriptors.rows >= 2) && (left_next_descriptors.rows >= 2)) {
            bf_matcher->knnMatch(left_descriptors, left_next_descriptors,knn_matches, 2);
        }
        else {
            std::cout << "not enough descriptors" << "\n";
            return -1;
        }
        const float ratio_thresh = 0.75f;
        std::vector<cv::DMatch> good_matches;
        for (size_t j = 0; j < knn_matches.size(); ++j) {
            if ((knn_matches[j].size() >= 2) 
                && (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)) {
                    
                good_matches.push_back(knn_matches[j][0]);
            }
        }

        std::vector<cv::Point3d> points_3d;
        std::vector<cv::Point2d> points_2d_next;

        for (const auto& match: good_matches) {
            cv::Point2f pt_curr = left_keypoints[match.queryIdx].pt;

            double u_float = static_cast<double>(pt_curr.x);
            double v_float = static_cast<double>(pt_curr.y);

            int u_int = static_cast<int>(pt_curr.x);
            int v_int = static_cast<int>(pt_curr.y);

            if ((u_int < 2)
                || (v_int < 2)
                || (u_int >= disp_float.cols - 2)
                || (v_int >= disp_float.rows - 2)) {
                    
                continue;
            }
            
            float disparity = disp_float.at<float>(v_int, u_int);
            double z = depth_map.at<double>(v_int, u_int);


            if ((z <= 0.5) || (z >= 300.0)) {
                continue;
            }

            double x = (u_float - cx) * z / fx;
            double y = (v_float - cy) * z / fy;
            points_3d.push_back(cv::Point3d(x, y, z));

            cv::Point2d pt_next = left_next_keypoints[match.trainIdx].pt;
            points_2d_next.push_back(pt_next);

        }

        if (points_3d.size() < 4) {
            std::cerr << "Frame " << i+1 << ": Not enough valid 3D-2D correspondences (" 
                      << points_3d.size() << "/" << good_matches.size() << " good matches), using previous pose" << "\n";
            trajectory_file << curr_pose_transform.at<double>(0, 0) << " "
                           << curr_pose_transform.at<double>(0, 1) << " "
                           << curr_pose_transform.at<double>(0, 2) << " "
                           << curr_pose_transform.at<double>(0, 3) << " "
                           << curr_pose_transform.at<double>(1, 0) << " "
                           << curr_pose_transform.at<double>(1, 1) << " "
                           << curr_pose_transform.at<double>(1, 2) << " "
                           << curr_pose_transform.at<double>(1, 3) << " "
                           << curr_pose_transform.at<double>(2, 0) << " "
                           << curr_pose_transform.at<double>(2, 1) << " "
                           << curr_pose_transform.at<double>(2, 2) << " "
                           << curr_pose_transform.at<double>(2, 3) << "\n";
            continue;
        }

        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        
        bool success = cv::solvePnPRansac(
            points_3d,
            points_2d_next,
            K,
            cv::Mat(),
            rvec,
            tvec,
            false,
            300,
            1.0,
            0.99,
            inliers,
            cv::SOLVEPNP_EPNP
        );

        if (!success || inliers.size() < 6 || (double)inliers.size() / points_3d.size() < 0.2) {
            std::cerr << "Frame " << i+1 << ": PnP failed or insufficient inliers (" 
                      << inliers.size() << "/" << points_3d.size() 
                      << ", ratio: " << (double)inliers.size() / points_3d.size() 
                      << "), using previous pose" << "\n";
            trajectory_file << curr_pose_transform.at<double>(0, 0) << " "
                           << curr_pose_transform.at<double>(0, 1) << " "
                           << curr_pose_transform.at<double>(0, 2) << " "
                           << curr_pose_transform.at<double>(0, 3) << " "
                           << curr_pose_transform.at<double>(1, 0) << " "
                           << curr_pose_transform.at<double>(1, 1) << " "
                           << curr_pose_transform.at<double>(1, 2) << " "
                           << curr_pose_transform.at<double>(1, 3) << " "
                           << curr_pose_transform.at<double>(2, 0) << " "
                           << curr_pose_transform.at<double>(2, 1) << " "
                           << curr_pose_transform.at<double>(2, 2) << " "
                           << curr_pose_transform.at<double>(2, 3) << "\n";
            continue;
        }

        if (inliers.size() >= 10) {
            std::vector<cv::Point3d> inlier_points_3d;
            std::vector<cv::Point2d> inlier_points_2d;
            for (int idx : inliers) {
                inlier_points_3d.push_back(points_3d[idx]);
                inlier_points_2d.push_back(points_2d_next[idx]);
            }
            
            cv::solvePnP(
                inlier_points_3d,
                inlier_points_2d,
                K,
                cv::Mat(),
                rvec,
                tvec,
                true,
                cv::SOLVEPNP_ITERATIVE
            );
        }

        if (((i + 1) % 100 == 0) || ((i+1) == left_images.size() - 1) || (i == 0)) {

            float percentage = (static_cast<float>(i+1)) / (left_images.size() - 1);
            int progress = static_cast<int>(percentage * 20);

            std::cout << "progress [";
            for(int bar_position = 0; bar_position < 20; bar_position++){
                if (bar_position < progress) std::cout << "=";
                else std::cout << " ";
            }
            std::cout << "]";
            std::cout << "Processed frame " << i+1 << "/" << left_images.size()-1 << "\r";
            std::cout.flush();
        }

        cv::Mat R;
        cv::Rodrigues(rvec, R);

        cv::Mat T_current_to_next = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R_64F, tvec_64F;
        R.convertTo(R_64F, CV_64F);
        tvec.convertTo(tvec_64F, CV_64F);
        R_64F.copyTo(T_current_to_next(cv::Rect(0, 0, 3, 3)));
        tvec_64F.copyTo(T_current_to_next(cv::Rect(3, 0, 1, 3)));
        
        cv::Mat T_next_to_current = T_current_to_next.inv();

        curr_pose_transform = curr_pose_transform * T_next_to_current;

        double tx = T_next_to_current.at<double>(0, 3);
        double ty = T_next_to_current.at<double>(1, 3);
        double tz = T_next_to_current.at<double>(2, 3);
        double translation_magnitude = std::sqrt(tx*tx + ty*ty + tz*tz);

        // format: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        trajectory_file << curr_pose_transform.at<double>(0, 0) << " "
                       << curr_pose_transform.at<double>(0, 1) << " "
                       << curr_pose_transform.at<double>(0, 2) << " "
                       << curr_pose_transform.at<double>(0, 3) << " "
                       << curr_pose_transform.at<double>(1, 0) << " "
                       << curr_pose_transform.at<double>(1, 1) << " "
                       << curr_pose_transform.at<double>(1, 2) << " "
                       << curr_pose_transform.at<double>(1, 3) << " "
                       << curr_pose_transform.at<double>(2, 0) << " "
                       << curr_pose_transform.at<double>(2, 1) << " "
                       << curr_pose_transform.at<double>(2, 2) << " "
                       << curr_pose_transform.at<double>(2, 3) << "\n";


    }

    trajectory_file.close();
    std::cout << "\nTrajectory saved to trajectory_" + sequence + ".txt" << "\n";

    return 0;
}