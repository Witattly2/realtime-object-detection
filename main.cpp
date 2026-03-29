#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class ObjectDetector {
public:
    ObjectDetector(std::string modelPath) {
        // Load model (simulated)
        std::cout << "Loading model from: " << modelPath << std::endl;
    }

    void processFrame(cv::Mat& frame) {
        // Simulated detection logic
        cv::rectangle(frame, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Object", cv::Point(50, 45), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }
};

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    ObjectDetector detector("yolov8.onnx");
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        detector.processFrame(frame);
        // cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}