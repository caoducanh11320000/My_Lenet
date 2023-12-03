#include <iostream>
#include "trt_inference.hpp"

static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int OUTPUT_SIZE = 10;

using namespace IMXAIEngine;
using namespace nvinfer1;

// static Logger gLogger;
TRT_Inference test1;

int main(int argc, char** argv){

    // create a model using the API directly and serialize it to a stream
    // char *trtModelStream{nullptr};
    // size_t size{0};
    //
    trt_error error = test1.init_inference(argc, argv);
    if (error == TRT_RESULT_ERROR)
    {
        return -1;
    }

    if(argc >=3){
        for (int i=2; i< argc ; i++){
        // Load and preprocess the image
        cv::Mat image = cv::imread(argv[i]);  // doc anh vao bang duong dan
        cv::resize(image, image, cv::Size(32, 32));  // Resize to match model input size
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);  // Convert to grayscale if needed
    
        float data[INPUT_H * INPUT_W];
        float prob[OUTPUT_SIZE];
        test1.trt_doInference(image, data, prob);
    }
    }

    return 1;
}