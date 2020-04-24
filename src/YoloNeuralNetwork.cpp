#include "YoloNeuralNetwork.hpp"
#include <fstream>

YoloNeuralNetwork::YoloNeuralNetwork(std::string &&yoloClassesFile,
        std::string &&yoloModelConfiguration, std::string && yoloModelWeights) :
        NeuralNetwork(cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights)),
        _yoloClassesFile(std::move(yoloClassesFile)),
        _yoloModelConfiguration(std::move(yoloModelConfiguration)),
        _yoloModelWeights(std::move(yoloModelWeights)),
        blobScaleFactor(1/255.0),
        blobSize(cv::Size(416, 416)),
        blobSwapRB(false),
        blobCrop(false)
{
    std::cout << "Yolo Neural Network constructor is called" << std::endl;
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<int> YoloNeuralNetwork::getOutputLayers()
{
    auto outLayers = _net.getUnconnectedOutLayers();
    return outLayers;
}

std::vector<std::string> YoloNeuralNetwork::getLayerNames()
{
    std::vector<cv::String> layersNames = _net.getLayerNames(); // get names of all layers in the network
    auto outLayers = getOutputLayers();
    std::vector<cv::String> names(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
    {
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

std::vector<std::string> YoloNeuralNetwork::getClassesNames()
{
    std::vector<std::string> classes;
    std::ifstream ifs(_yoloClassesFile.c_str());
    std::string line;
    while (getline(ifs, line)) 
    {
        classes.push_back(line);
    }
    return classes;
}

std::vector<cv::Mat> YoloNeuralNetwork::processInputImg(const cv::Mat& img)
{
    cv::Mat blob = generateBlobFromInputImg(img);

    // Get names of output layers
    std::vector<cv::String> names = getLayerNames();

    std::vector<cv::Mat> netOutput;
    _net.setInput(blob);
    _net.forward(netOutput, names);

    return netOutput;
}

std::vector<cv::Mat> YoloNeuralNetwork::processInputImg(std::string&& inputImagePath)
{
    cv::Mat img = cv::imread(inputImagePath);

    return processInputImg(img);
}

cv::Mat YoloNeuralNetwork::generateBlobFromInputImg(const cv::Mat& img)
{
    cv::Mat blob;
    cv::Scalar mean = cv::Scalar(0,0,0);

    cv::dnn::blobFromImage(img, blob, blobScaleFactor, blobSize, mean, blobSwapRB, blobCrop);

    return blob;
}


std::vector<YoloNeuralNetwork::BoundingBox> YoloNeuralNetwork::getBoundingBoxes(
    const cv::Mat& img, float confThreshold, float nmsThreshold)
{
    std::vector<cv::Mat> netOutput = processInputImg(img);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;
            
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top
                
                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    // perform non-maxima suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it)
    {
        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box
        
        bBoxes.push_back(bBox);
    }
    return bBoxes;
}

cv::Mat YoloNeuralNetwork::drawBoundingBoxes(const cv::Mat & img, const std::vector<YoloNeuralNetwork::BoundingBox>& bBoxes)
{
    std::vector<std::string> classes = getClassesNames();
    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it)
    {
        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("%.2f", (*it).confidence);
        label = classes[((*it).classID)] + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
        top = std::max(top, labelSize.height);
        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 1);
    }
    return visImg;
}
