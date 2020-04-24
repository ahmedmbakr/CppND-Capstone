# CPP_ND_CAPSTONE README

This project classifies an input image and draws a new image that shows the detected outputs inside the image and it also shows the classification percentage on top of the drawn rectangle.
The classification is done using [Yolo](https://pjreddie.com/darknet/yolo/) pretrained network.

The project contains the following files:

- `NeuralNetwork.hpp` which is a virtual abstract class that define the important asspects of any neural network
- `YoloNeuralNetwork.cpp` which inherits from `NeuralNetwork` class and implements its virtual functions to classify the input image, and generate the output classifications as bounding boxes and it also produce APIs to draw the output image with bounding boxes in it
- `detect_objects.cpp` The main file that runs the whole program and contains the main function. You have to edit the first line that reads image path to see the output of different images

All functions are documented in Doxygent format so we can generate a documentation for our library so that it can be used by other users.

For testing, `img` folder contains multiple images to try the classifier and its power in classifying objects.

## How to run the program

- `git clone https://github.com/ahmedmbakr/CppND-Capstone.git`
- `wget https://pjreddie.com/media/files/yolov3.weights`
- `mkdir build && cd build`
- `cmake ..`
- `make && ./yolo_detect_objects`
