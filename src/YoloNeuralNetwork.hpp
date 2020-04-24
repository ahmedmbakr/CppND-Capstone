#pragma once

#include "NeuralNetwork.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

class YoloNeuralNetwork : public NeuralNetwork<cv::dnn::Net, cv::Mat>
{
    public:
    /**
     * bounding box around a classified object (contains both 2D and 3D data)
     */
    struct BoundingBox { 
        /** unique identifier for this bounding box*/
        int boxID;
        /** 2D region-of-interest in image coordinates*/
        cv::Rect roi;
        /** ID based on class file provided to YOLO framework*/
        int classID;
        /** classification trust*/
        double confidence;
    };

    std::vector<std::string> getLayerNames() override;

    std::vector<cv::Mat> processInputImg(std::string&& inputImagePath) override;

    /**
     * Process input image and generate the output classifications as a vector
     */
    std::vector<cv::Mat> processInputImg(const cv::Mat& inputImage);

    /**
     * Get the classes names from the file whose path in the member \ref{yoloClassesFile}
    */
    std::vector<std::string> getClassesNames();

    cv::Mat drawBoundingBoxes(const cv::Mat & inputImg, const std::vector<BoundingBox>& bBoxes);

    /**
     * Process input image by its path and generate the bounding boxes
     * @param[in] inputImage Input image
     * @param[in] confThreshold Confidece threshold
     * @param[in] nmsThreshold non mamxima supression
     * @param[out] bBoxes vector of bounding boxes
     */ 
    std::vector<BoundingBox> getBoundingBoxes(const cv::Mat& inputImage, float confThreshold, float nmsThreshold);

    YoloNeuralNetwork(std::string &&yoloClassesFile,
        std::string &&yoloModelConfiguration, std::string && yoloModelWeights);

    ~YoloNeuralNetwork()
    {
        std::cout << "YoloNeuralNetwork destructor is called" << std::endl;
    }

    double blobScaleFactor;
    cv::Size blobSize;
    bool blobSwapRB;
    bool blobCrop;
    protected:

    private:
    /**
     * Get indices of output layers, i.e. layers with unconnected outputs
    */
    std::vector<int> getOutputLayers();

    cv::Mat generateBlobFromInputImg(const cv::Mat& img);

    std::string _yoloBasePath;
    std::string _yoloClassesFile;
    std::string _yoloModelConfiguration;
    std::string _yoloModelWeights; 

};