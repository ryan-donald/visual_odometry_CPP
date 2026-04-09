#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

class BoWDatabase
{

public:
    int vocabulary_size;
    bool is_trained;
    cv::Mat vocabulary;
    std::vector<cv::Mat> keyframe_vectors;
    std::unordered_map<int, std::unordered_set<int>> inverted_index;

    BoWDatabase()
        : vocabulary_size(1000),
          is_trained(false) {

          };

    BoWDatabase(int vocabulary_size)
        : vocabulary_size(vocabulary_size),
          is_trained(false) {

          };

    void train_vocabulary(const std::vector<cv::Mat> &all_descriptors)
    {
        cv::Mat all_desc_float;

        for (const auto &d : all_descriptors)
        {
            cv::Mat d32;
            d.convertTo(d32, CV_32F);
            all_desc_float.push_back(d32);
        }

        if (all_desc_float.rows < vocabulary_size)
        {
            vocabulary_size = std::max(1, all_desc_float.rows / 2);
        }

        cv::Mat labels;
        cv::Mat centers;
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.001);
        int attempts = 3;
        cv::kmeans(all_desc_float, vocabulary_size, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);

        this->vocabulary = centers.clone();
        this->is_trained = true;
    };

    cv::Mat compute_bow_vector(const cv::Mat &descriptors)
    {
        cv::Mat bow = cv::Mat::zeros(1, vocabulary_size, CV_32F);
        if (!is_trained || descriptors.empty())
            return bow;
        cv::Mat vocab32;
        if (vocabulary.type() != CV_32F)
            vocabulary.convertTo(vocab32, CV_32F);
        else
            vocab32 = vocabulary;

        cv::Mat desc32;
        descriptors.convertTo(desc32, CV_32F);

        for (int i = 0; i < desc32.rows; ++i)
        {
            cv::Mat drow = desc32.row(i);
            cv::Mat diffs = vocab32 - cv::repeat(drow, vocab32.rows, 1);
            cv::Mat sq;
            cv::multiply(diffs, diffs, sq);
            cv::Mat dists;
            cv::reduce(sq, dists, 1, cv::REDUCE_SUM, CV_32F);

            double minVal;
            cv::Point minLoc;
            cv::minMaxLoc(dists, &minVal, nullptr, &minLoc, nullptr);
            int word_id = minLoc.y;
            bow.at<float>(0, word_id) += 1.0f;
        }

        double s = cv::sum(bow)[0];
        if (s > 0.0)
            bow /= static_cast<float>(s);
        cv::normalize(bow, bow, 1.0, 0.0, cv::NORM_L2);
        return bow;
    }

    void add_keyframe(const KeyFrame &kf)
    {
        if (!is_trained)
            return;
        cv::Mat bow = compute_bow_vector(kf.descriptors);
        keyframe_vectors[kf.frame_id] = bow.clone();
        for (size_t i = 0; i < bow.cols; i++)
        {
            if (bow.at<float>(0, i) > 0.0f)
            {
                inverted_index[i].insert(kf.frame_id);
            }
        }
    }

private:
};

class KeyFrame
{
public:
    int frame_id;
    cv::Mat pose_transform;
    cv::Mat left_image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat relative_pose_from_prev;
    int num_loop_closures;
    float uncertainty;
    cv::Point3d position;
    cv::Mat bow_vector;

    KeyFrame(int frame_id, const cv::Mat &left_image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors,
             const cv::Mat &pose_transform, const cv::Mat &relative_pose_from_prev)
        : frame_id(frame_id),
          left_image(left_image),
          keypoints(keypoints),
          descriptors(descriptors),
          pose_transform(pose_transform),
          relative_pose_from_prev(relative_pose_from_prev),
          num_loop_closures(0),
          uncertainty(1.0),
          position(pose_transform.at<double>(0, 3), pose_transform.at<double>(1, 3), pose_transform.at<double>(2, 3))
    {
    }

    void setBowVector(cv::Mat bow_vector)
    {
        this->bow_vector = bow_vector;
    }
};
