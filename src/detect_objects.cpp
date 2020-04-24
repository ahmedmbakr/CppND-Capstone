#include <iostream>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "YoloNeuralNetwork.hpp"

using namespace std;

void detectObjects()
{
    // load image from file
    cv::Mat img = cv::imread("../images/s_thrun.jpg");

    // load class names from file
    string yoloBasePath = "../dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights"; 

    YoloNeuralNetwork yoloNeuralNetwork(std::move(yoloClassesFile),
    std::move(yoloModelConfiguration), std::move(yoloModelWeights));

    vector<string> classes = yoloNeuralNetwork.getClassesNames();
    float confThreshold = 0.20;
    float nmsThreshold = 0.40;
    std::vector<YoloNeuralNetwork::BoundingBox> bBoxes = yoloNeuralNetwork.getBoundingBoxes(img, confThreshold, nmsThreshold);
    
    cv::Mat visImg = yoloNeuralNetwork.drawBoundingBoxes(img, bBoxes);

    string windowName = "Object classification";
    cv::namedWindow( windowName, 1 );
    cv::imshow( windowName, visImg );
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    detectObjects();
}