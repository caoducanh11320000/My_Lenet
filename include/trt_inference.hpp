#ifndef TRT_INFERENCE_HPP
#define TRT_INFERENCE_HPP

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace nvinfer1;

namespace IMXAIEngine
{

    typedef enum{
        TRT_RESULT_SUCCESS,
        TRT_RESULT_ERROR
    } trt_error;


    typedef struct
    {
            
    } trt_results;
    

    class TRT_Inference
    {
    private:
        IRuntime* runtime = NULL;
        ICudaEngine* engine = NULL;
        IExecutionContext* context = NULL;
        
    public:
        TRT_Inference();
        ~TRT_Inference();
        trt_error init_inference(int argc, char **argv);
        //void trt_APIModel(unsigned int maxBatchSize, IHostMemory** modelStream);
        trt_error trt_detection(std::vector<cv::Mat> &input_img, std::vector<trt_results> &results);
        trt_error test();
        trt_error trt_doInference(cv::Mat& input_img, float* data, float* prob);
    };

} // namespace IMXAIEngine

#endif