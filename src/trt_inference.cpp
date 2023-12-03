#include "trt_inference.hpp"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

// phan nay co the bo vao thu vien, la cac tinh chat cua OOP
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static Logger gLogger;

using namespace IMXAIEngine;
using namespace nvinfer1;

TRT_Inference::TRT_Inference(){
printf("Khai bao doi tuong thanh cong\n");
}

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

ICudaEngine* createLenetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
    assert(data);

    // Add convolution layer with 6 outputs and a 5x5 filter.
    std::map<std::string, Weights> weightMap = loadWeights("../lenet5.wts");
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 6, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    // Add second convolution layer with 16 outputs and a 5x5 filter.
    IConvolutionLayer* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 16, DimsHW{5, 5}, weightMap["conv2.weight"], weightMap["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x2>
    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});

    // Add fully connected layer
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 120, weightMap["fc1.weight"], weightMap["fc1.bias"]);
    assert(fc1);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu3 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    // Add second fully connected layer
    IFullyConnectedLayer* fc2 = network->addFullyConnected(*relu3->getOutput(0), 84, weightMap["fc2.weight"], weightMap["fc2.bias"]);
    assert(fc2);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu4 = network->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    // Add third fully connected layer
    IFullyConnectedLayer* fc3 = network->addFullyConnected(*relu4->getOutput(0), OUTPUT_SIZE, weightMap["fc3.weight"], weightMap["fc3.bias"]);
    assert(fc3);

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*fc3->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void trt_APIModel(unsigned int maxBatchSize, IHostMemory** modelStream){
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createLenetEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

trt_error TRT_Inference::trt_detection(std::vector<cv::Mat> &input_img, std::vector<trt_results> &results){
    std::cout <<"Hello" << std::endl;
return TRT_RESULT_SUCCESS;   
}

trt_error TRT_Inference::test(){
    std::cout <<"Hello" << std::endl;
return TRT_RESULT_SUCCESS;  
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float))); // bo check di duoc ko
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

trt_error TRT_Inference::init_inference(int argc, char **argv){
    if (argc <2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./lenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./lenet -d   // deserialize plan file and run inference" << std::endl;
        return TRT_RESULT_ERROR;
    }
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        trt_APIModel(1, &modelStream); //ham tao engine
        assert(modelStream != nullptr);

        std::ofstream p("lenet5.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return TRT_RESULT_ERROR;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return TRT_RESULT_SUCCESS;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("lenet5.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return TRT_RESULT_ERROR;
    }

    // // Load and preprocess the image
    // cv::Mat image = cv::imread(argv[2]);  // doc anh vao bang duong dan
    // cv::resize(image, image, cv::Size(32, 32));  // Resize to match model input size
    // cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);  // Convert to grayscale if needed
    //// Goi ham do Inference tu day
    // Convert the image data to the format expected by the model
    // float data[INPUT_H * INPUT_W];
    // for (int i = 0; i < 32; ++i) {
    //     for (int j = 0; j < 32; ++j) {
    //         data[i * 32 + j] = static_cast<float>(image.at<uchar>(i, j)) / 255.0;
    //     }
    // }
    // phan nay nen de tron ham Init ko ?
    // chac ko can, vi cac bien nay ko the cos gia tri tu ban dau, nhung co the cho thanh Attribute
    
    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(this->engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(this->context != nullptr);
    ////////

    // Run inference
    // Convert the image data to the format expected by the model

    // float data[INPUT_H * INPUT_W];
    // for (int i = 0; i < 32; ++i) {
    //     for (int j = 0; j < 32; ++j) {
    //         data[i * 32 + j] = static_cast<float>(image.at<uchar>(i, j)) / 255.0;
    //     }
    // }
    // float prob[OUTPUT_SIZE];
    // for (int i = 0; i < 1000; i++) {
    //     auto start = std::chrono::system_clock::now();
    //     doInference(*context, data, prob, 1);
    //     auto end = std::chrono::system_clock::now();
    //     //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // }

    // Destroy the engine  
    /// co the bo cac lenh nay vao ham Destructor
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();

    // Print histogram of the output distribution
    // std::cout << "\nOutput:\n\n";
    // for (unsigned int i = 0; i < 10; i++)
    // {
    //     std::cout << prob[i] << ", ";
    // }
    // std::cout << std::endl;

    return TRT_RESULT_SUCCESS;
}

trt_error TRT_Inference::trt_doInference(cv::Mat& input_img, float* data, float* prob){
    for (int i = 0; i < INPUT_H; ++i) {
        for (int j = 0; j < INPUT_W; ++j) {
            data[i *INPUT_H + j] = static_cast<float>(input_img.at<uchar>(i, j)) / 255.0;
        }
    }
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;

    return TRT_RESULT_SUCCESS;
}
TRT_Inference::~TRT_Inference(){
if (context) {
    context->destroy();
    //context = nullptr;  // Gán giá trị nullptr sau khi giải phóng để tránh double delete
}
if (engine) {
    engine->destroy();
   // engine = nullptr;
}
if (runtime) {
    runtime->destroy();
    //runtime = nullptr;
    }
printf("Doi tuong da dc huy\n");
}