#include "caffe/layers/Pooling/pooling_layer_cudnn.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDNNPoolingLayer<Dtype>::InitLayer() {
    if (!cudnn_handle_) {
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
      CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc_));
    }
  }

  template <typename Dtype>
  CUDNNPoolingLayer<Dtype>::~CUDNNPoolingLayer() {
    if (cudnn_handle_) {
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
      CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pooling_desc_));
    }
  }

  template <typename Dtype>
  void CUDNNPoolingLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    this->CheckMode();
    
    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    std::vector<int> output_shape;
    this->ComputeOutputShape(input_shape, output_shape);
  
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   input_shape[0], //n
					   input_shape[1], //c
					   input_shape[2], //h
					   input_shape[3]  //w
					   ));
    
    cudnnPoolingMode_t cudnn_pooling_mode;
    if (this->mode_ == POOLING_MAX){
      cudnn_pooling_mode = CUDNN_POOLING_MAX;
    }else if (this->mode_ == POOLING_AVE){
      cudnn_pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_desc_,
					    cudnn_pooling_mode,
					    CUDNN_PROPAGATE_NAN,
					    this->kernel_size_h_,
					    this->kernel_size_w_,
					    this->pad_h_,
					    this->pad_w_,
					    this->stride_h_,
					    this->stride_w_
					    ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   output_shape[0], //n
					   output_shape[1], //c
					   output_shape[2], //h
					   output_shape[3]  //w
					   ));

    this->outputs_[0]->Reshape(output_shape);
  }

  template <typename Dtype>
  void CUDNNPoolingLayer<Dtype>::ForwardComputation() {
    Dtype alpha = Dtype(1);
    Dtype beta  = Dtype(0);
    const Dtype * input_data = this->inputs_[0]->gpu_data();
    Dtype * output_data = this->outputs_[0]->mutable_gpu_data();

    CHECK_CUDNN(cudnnPoolingForward(cudnn_handle_,
				    pooling_desc_,
				    &alpha,
				    input_desc_,
				    input_data,
				    &beta,
				    output_desc_,
				    output_data
				    ));
  }

  INSTANTIATE_CLASS(CUDNNPoolingLayer);

} // namespace facethink
