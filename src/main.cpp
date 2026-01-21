#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;
    
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cam" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "blank frame" << std::endl;
            break;
        }

        cv::imshow("webcam frames", frame);

        int key = cv::waitKey(1);

        if (key == 'q' || key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}