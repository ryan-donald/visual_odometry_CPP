#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

// struct for calibration information.
struct CameraCalibration {
    float fx, fy, cx, cy;
    float baseline;
    cv::Mat K;
};

// struct for depth and disparity maps.
struct StereoResult {
    cv::Mat depth_map;
    cv::Mat disparity_map;
};

// struct for feature matching.
struct FeatureMatches {
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point2d> points_2d_next;
    size_t num_good_matches;
};

// parses the configuration file for the given image.
bool parse_configuration(const std::string& filepath, float left_image_calibration[3][4], float right_image_calibration[3][4]) {
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open calibration file: " << filepath << "\n";
        return false;
    }
    
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string label;
        iss >> label;

        if (label.empty()) continue;

        float* target_matrix = nullptr;

        if (label == "P0:") {
            target_matrix = &left_image_calibration[0][0];
        } else if (label == "P1:") {
            target_matrix = &right_image_calibration[0][0];
        } else {
            break;
        }

        for (int i = 0; i < 12; ++i) {
            if (!(iss >> target_matrix[i])) {
                std::cerr << "Error reading value " << i << " from " << label << "\n";
                return false;
            }
        }
    }
    
    file.close();
    return true;
}

// load the images for a given sequence.
bool load_image_paths(const std::string& sequence, std::vector<cv::String>& left_images, std::vector<cv::String>& right_images) {
    std::string left_directory_path = "../../../ryan_slam/kitti/sequences/" + sequence + "/image_0/*.png";
    std::string right_directory_path = "../../../ryan_slam/kitti/sequences/" + sequence + "/image_1/*.png";
    
    cv::glob(left_directory_path, left_images, false);
    cv::glob(right_directory_path, right_images, false);

    if (left_images.empty() || right_images.empty()) {
        std::cerr << "No images found for sequence " << sequence << "\n";
        return false;
    }
    return true;
}

// initializes camera calibration struct.
CameraCalibration initialize_camera_calibration(const float left_image_calibration[3][4], 
                                                  const float right_image_calibration[3][4]) {
    CameraCalibration calib;
    calib.fx = left_image_calibration[0][0];
    calib.fy = left_image_calibration[1][1];
    calib.cx = left_image_calibration[0][2];
    calib.cy = left_image_calibration[1][2];
    calib.baseline = std::abs(left_image_calibration[0][3] - right_image_calibration[0][3]) / calib.fx;
    
    calib.K = (cv::Mat_<double>(3, 3) << 
        calib.fx, 0, calib.cx,
        0, calib.fy, calib.cy,
        0, 0, 1
    );
    
    return calib;
}

// create stereo matcher for disparity map.
cv::Ptr<cv::StereoSGBM> create_stereo_matcher() {
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

    return cv::StereoSGBM::create(
        minDisparity, numDisparities, blockSize,
        P1, P2, disp12MaxDiff, preFilterCap,
        uniquenessRatio, speckleWindowSize, speckleRange,
        cv::StereoSGBM::MODE_SGBM_3WAY
    );
}

// create ORB feature detector for detecting features and descriptors in an image.
cv::Ptr<cv::ORB> create_feature_detector() {
    int nFeatures = 6000;
    float scaleFactor = 1.2;
    int nLevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    int fastThreshold = 15;

    return cv::ORB::create(
        nFeatures, scaleFactor, nLevels,
        edgeThreshold, firstLevel, WTA_K,
        cv::ORB::HARRIS_SCORE, patchSize, fastThreshold
    );
}

// compute stereo disparity and depth maps.
StereoResult compute_stereo_depth(const cv::Mat& left_gray, const cv::Mat& right_gray,
                                   cv::Ptr<cv::StereoSGBM> matcher, const CameraCalibration& calib) {
    StereoResult result;
    
    cv::Mat disp_map;
    matcher->compute(left_gray, right_gray, disp_map);
    
    cv::Mat disp_float;
    disp_map.convertTo(disp_float, CV_32F, 1.0/16.0);
    result.disparity_map = disp_float;
    
    result.depth_map = cv::Mat::zeros(disp_float.size(), CV_64F);
    for (int row = 0; row < disp_float.rows; ++row) {
        for (int col = 0; col < disp_float.cols; ++col) {
            float disp = disp_float.at<float>(row, col);
            result.depth_map.at<double>(row, col) = (calib.fx * calib.baseline) / (disp + 1e-6);
        }
    }
    
    return result;
}

// detect features temporally in two images from consecutive timesteps.
FeatureMatches match_features_and_reconstruct_3d(
    const cv::Mat& left_gray, const cv::Mat& next_left_gray,
    const StereoResult& stereo, const CameraCalibration& calib,
    cv::Ptr<cv::ORB> detector, cv::Ptr<cv::BFMatcher> matcher) {
    
    FeatureMatches result;
    result.num_good_matches = 0;
    
    // detect keypoints and descriptors.
    std::vector<cv::KeyPoint> left_keypoints, left_next_keypoints;
    cv::Mat left_descriptors, left_next_descriptors;
    
    detector->detectAndCompute(left_gray, cv::noArray(), left_keypoints, left_descriptors);
    detector->detectAndCompute(next_left_gray, cv::noArray(), left_next_keypoints, left_next_descriptors);
    
    if (left_descriptors.rows < 2 || left_next_descriptors.rows < 2) {
        return result;
    }
    
    // KNN matching between the two frames.
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(left_descriptors, left_next_descriptors, knn_matches, 2);
    
    // ratio test.
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t j = 0; j < knn_matches.size(); ++j) {
        if (knn_matches[j].size() >= 2 && 
            knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance) {
            good_matches.push_back(knn_matches[j][0]);
        }
    }
    
    result.num_good_matches = good_matches.size();
    
    // create 3-D points from depth map for good matches.
    for (const auto& match : good_matches) {
        cv::Point2f pt_curr = left_keypoints[match.queryIdx].pt;
        
        double u_float = static_cast<double>(pt_curr.x);
        double v_float = static_cast<double>(pt_curr.y);
        int u_int = static_cast<int>(pt_curr.x);
        int v_int = static_cast<int>(pt_curr.y);
        
        // boundary check.
        if (u_int < 2 || v_int < 2 || 
            u_int >= stereo.disparity_map.cols - 2 || 
            v_int >= stereo.disparity_map.rows - 2) {
            continue;
        }
        
        double z = stereo.depth_map.at<double>(v_int, u_int);
        
        // valid depth check.
        if (z <= 0.5 || z >= 300.0) {
            continue;
        }
        
        // reconstruct 3-D point.
        double x = (u_float - calib.cx) * z / calib.fx;
        double y = (v_float - calib.cy) * z / calib.fy;
        result.points_3d.push_back(cv::Point3d(x, y, z));
        
        cv::Point2d pt_next = left_next_keypoints[match.trainIdx].pt;
        result.points_2d_next.push_back(pt_next);
    }
    
    return result;
}

// use Perspective-n-Point to solve transformation between frames.
bool estimate_pose(const FeatureMatches& matches, const cv::Mat& K,
                   cv::Mat& rvec, cv::Mat& tvec, std::vector<int>& inliers) {
    
    if (matches.points_3d.size() < 4) {
        return false;
    }
    
    bool success = cv::solvePnPRansac(
        matches.points_3d, matches.points_2d_next, K, cv::Mat(),
        rvec, tvec, false, 300, 1.0, 0.99, inliers, cv::SOLVEPNP_EPNP
    );
    
    if (!success || inliers.size() < 6 || 
        (double)inliers.size() / matches.points_3d.size() < 0.2) {
        return false;
    }
    
    // refine with iterative method if enough inliers
    if (inliers.size() >= 10) {
        std::vector<cv::Point3d> inlier_points_3d;
        std::vector<cv::Point2d> inlier_points_2d;
        for (int idx : inliers) {
            inlier_points_3d.push_back(matches.points_3d[idx]);
            inlier_points_2d.push_back(matches.points_2d_next[idx]);
        }
        
        cv::solvePnP(inlier_points_3d, inlier_points_2d, K, cv::Mat(),
                     rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    }
    
    return true;
}

// update estimated trajectory by accumulating transformations.
void update_pose(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& curr_pose_transform) {
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
}

// add new pose to trajectory.
void write_pose_to_file(std::ofstream& file, const cv::Mat& pose) {
    file << pose.at<double>(0, 0) << " " << pose.at<double>(0, 1) << " "
         << pose.at<double>(0, 2) << " " << pose.at<double>(0, 3) << " "
         << pose.at<double>(1, 0) << " " << pose.at<double>(1, 1) << " "
         << pose.at<double>(1, 2) << " " << pose.at<double>(1, 3) << " "
         << pose.at<double>(2, 0) << " " << pose.at<double>(2, 1) << " "
         << pose.at<double>(2, 2) << " " << pose.at<double>(2, 3) << "\n";
}

// display progress bar in terminal for user.
void display_progress(size_t current, size_t total) {
    if ((current % 100 == 0) || (current == total) || (current == 1)) {
        float percentage = static_cast<float>(current) / total;
        int progress = static_cast<int>(percentage * 20);
        
        std::cout << "progress [";
        for (int bar_position = 0; bar_position < 20; bar_position++) {
            if (bar_position < progress) std::cout << "=";
            else std::cout << " ";
        }
        std::cout << "] Processed frame " << current << "/" << total << "\r";
        std::cout.flush();
    }
}

int main() {
    // prompt the user for input and store the result in sequence.
    std::cout << "input sequence to be utilized: ";
    std::string sequence;
    std::cin >> sequence;

    // load image paths for the provided sequence.
    std::vector<cv::String> left_images, right_images;
    if (!load_image_paths(sequence, left_images, right_images)) {
        return -1;
    }

    // parse the camera calibration file for the specific sequence.
    std::string camera_calib_path = "../../../ryan_slam/kitti/sequences/" + sequence + "/calib.txt";
    float left_image_calibration[3][4];
    float right_image_calibration[3][4];

    if (!parse_configuration(camera_calib_path, left_image_calibration, right_image_calibration)) {
        std::cerr << "parsing calibration file failed" << "\n";
        return -1;
    }

    // initialize camera calibration parameters.
    CameraCalibration calib = initialize_camera_calibration(left_image_calibration, right_image_calibration);
    cv::Mat curr_pose_transform = cv::Mat::eye(4, 4, CV_64F);

    // create objects for image processing.
    cv::Ptr<cv::StereoSGBM> stereo_matcher = create_stereo_matcher();
    cv::Ptr<cv::ORB> feature_detector = create_feature_detector();
    cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

    // initialize the output trajectory file.
    std::ofstream trajectory_file("trajectory_" + sequence + ".txt");
    if (!trajectory_file.is_open()) {
        std::cerr << "Failed to open trajectory.txt for writing" << "\n";
        return -1;
    }

    trajectory_file << "1 0 0 0 0 1 0 0 0 0 1 0" << "\n";

    std::cout << "starting visual odometry calculation for sequence " 
              << sequence << " with " << left_images.size() - 1 << " frame pairs" << "\n";

    // main loop that processes each frame pair.
    for (size_t i = 0; i < left_images.size() - 1; i++) {
        // load the three images needed (stereo pair and left image from next timestep).
        cv::Mat left_image = cv::imread(left_images[i]);
        cv::Mat right_image = cv::imread(right_images[i]);
        cv::Mat next_left_image = cv::imread(left_images[i+1]);

        if (left_image.empty() || right_image.empty() || next_left_image.empty()) {
            std::cerr << "Failed to load images at frame " << i << "\n";
            return -1;
        }

        // convert images to grayscale.
        cv::Mat left_gray, right_gray, next_left_gray;
        cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(next_left_image, next_left_gray, cv::COLOR_BGR2GRAY);

        // compute depth map and create 2-d and 3-d points.
        StereoResult stereo = compute_stereo_depth(left_gray, right_gray, stereo_matcher, calib);
        FeatureMatches matches = match_features_and_reconstruct_3d(
            left_gray, next_left_gray, stereo, calib, feature_detector, bf_matcher
        );

        // check if we have enough points for PnP.
        if (matches.points_3d.size() < 4) {
            std::cerr << "Frame " << i+1 << ": Not enough valid 3D-2D correspondences (" 
                      << matches.points_3d.size() << "/" << matches.num_good_matches 
                      << " good matches), using previous pose" << "\n";
            write_pose_to_file(trajectory_file, curr_pose_transform);
            continue;
        }

        // estimate transformation between timesteps using PnP.
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        
        if (!estimate_pose(matches, calib.K, rvec, tvec, inliers)) {
            std::cerr << "Frame " << i+1 << ": PnP failed or insufficient inliers, using previous pose" << "\n";
            write_pose_to_file(trajectory_file, curr_pose_transform);
            continue;
        }

        // update cumulative pose transformation.
        update_pose(rvec, tvec, curr_pose_transform);

        // write pose to trajectory file.
        write_pose_to_file(trajectory_file, curr_pose_transform);

        // display progress.
        display_progress(i + 1, left_images.size() - 1);
    }

    // close output file and notify user.
    trajectory_file.close();
    std::cout << "\nTrajectory saved to trajectory_" + sequence + ".txt" << "\n";

    return 0;
}