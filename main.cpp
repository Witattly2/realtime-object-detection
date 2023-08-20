#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Function to load YOLO model
cv::dnn::Net load_yolo_model(const std::string& model_cfg, const std::string& model_weights)
{
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(model_cfg, model_weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return net;
}

// Function to get output layer names
std::vector<std::string> get_output_names(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        // Get the names of all the layers in the network
        std::vector<int> out_layers = net.getUnconnectedOutLayers();
        std::vector<std::string> layers_names = net.getLayerNames();

        // Get the names of the output layers in the network
        names.resize(out_layers.size());
        for (size_t i = 0; i < out_layers.size(); ++i)
            names[i] = layers_names[out_layers[i] - 1];
    }
    return names;
}

// Function to draw bounding boxes and labels
void draw_pred(int class_id, float conf, int left, int top, int right, int bottom, cv::Mat& frame, const std::vector<std::string>& classes)
{
    // Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);

    // Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(class_id < (int)classes.size());
        label = classes[class_id] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int base_line;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
    top = std::max(top, label_size.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*label_size.height)), cv::Point(left + round(1.5*label_size.width), top + base_line), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}

// Main detection function
void post_process(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<std::string>& classes, float conf_threshold, float nms_threshold)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores.
        // Assign the class label to the bounding box object
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point class_id_point;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > conf_threshold)
            {
                int center_x = (int)(data[0] * frame.cols);
                int center_y = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = center_x - width / 2;
                int top = center_y - height / 2;

                class_ids.push_back(class_id_point.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        draw_pred(class_ids[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);
    }
}

int main()
{
    // Load class names
    std::vector<std::string> classes;
    std::ifstream ifs("coco.names");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    // Load the network
    cv::dnn::Net net = load_yolo_model("yolov3.cfg", "yolov3.weights");

    // Open a video file or an image file or a camera stream.
    std::string source = "test_video.mp4"; // Replace with 0 for webcam, or an image file
    cv::VideoCapture cap(source);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video source." << std::endl;
        return -1;
    }

    cv::Mat frame, blob;
    float conf_threshold = 0.5; // Confidence threshold
    float nms_threshold = 0.4;  // Non-maximum suppression threshold
    int input_width = 416;      // Width of network\'s input image
    int input_height = 416;     // Height of network\'s input image

    while (cv::waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cv::waitKey(3000);
            break;
        }

        // Create a 4D blob from a frame.
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(input_width, input_height), cv::Scalar(0,0,0), true, false);
        
        // Set the input to the network
        net.setInput(blob);

        // Run the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;
        net.forward(outs, get_output_names(net));

        // Remove the bounding boxes with low confidence
        post_process(frame, outs, classes, conf_threshold, nms_threshold);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("FPS: %.2f", 1000/t);
        cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

        // Show the image
        cv::imshow("Real-time Object Detection", frame);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
