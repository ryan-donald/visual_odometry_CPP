#include <opencv2/opencv.hpp>
#include <vector>
class LoopClosureDetector
{

public:
    int min_matches;
    int temporal_gap;
    bool use_bow;
    bool use_spatial;
    double bow_min_score;
    double spatial_radius;

    bool train_bow_vocabulary(std::vector<cv::Mat> descriptors)
    {
    }

private:
};