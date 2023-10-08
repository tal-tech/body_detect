# Project: Caffe inference in Windows.

## Object: deploy caffemodel with minimum external dependencies.

### How to compile it:

1. Required Libraries: Boost, OpenCV, protobuf.

2. A CMake option **CPU_ONLY** is set to indicate on which platform (CPU or GPU) we want to deploy our model. (ON/OFF)


so, if you want a SSD caffe inference on CPU (default CPU_ONLY=OFF), just run:

    cmake -DCPU_ONLY=ON ..       


### How to use it:

a sample code:

    #include "caffe/caffe.hpp"
    int main() {
    const std::string prototxt = "face_angle_deploy.prototxt";
    const std::string model_file = "face_angle.caffemodel";
    facethink::NetBuilder<float> net_builder;
    auto net = net_builder.Create(prototxt, model_file);
    
    // input blob
    auto blob_data = net->blob("data");
    
    // for example:
    // float* fdata: a point to input image data
    // std::vector<int> fdata_shape: a vector that stores input image shape (60*60)
    blob_data->Reshape(fdata_shape);
    blob_data->ImportFromExtenalData(fdata, 60*60);   //(point, number of elements)
    net->Forward();
    
    auto blob_output = net->blob("Dense3");
    const float* data_output = blob_output->cpu_data(); // point to the result
    int count_output = blob_output->count(); // number of elements.
    return 0;
    }

### Test data download

The test data can be found in ftp://10.1.2.202/mirror/Caffe_Inference_SDK/base

The username and passward to login ftp server: https://redmine.facethink.com/documents/4