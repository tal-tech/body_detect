#include "caffe/layers/LRN/lrn_layer_cudnn.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDNNLRNLayer<Dtype>::InitLayer() {
    if (!cudnn_handle_) {
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
      CHECK_CUDNN(cudnnCreateLRNDescriptor(&lrn_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
    }
  }

  template <typename Dtype>
  CUDNNLRNLayer<Dtype>::~CUDNNLRNLayer() {
    if (cudnn_handle_) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
      CHECK_CUDNN(cudnnDestroyLRNDescriptor(lrn_desc_));
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
    }
  }

  template <typename Dtype>
  void CUDNNLRNLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int> input_shape = this->inputs_[0]->shape();    
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   input_shape[0], //n
					   input_shape[1], //c
					   input_shape[2], //h
					   input_shape[3]  //w
					   ));


    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   input_shape[0], //n
					   input_shape[1], //c
					   input_shape[2], //h
					   input_shape[3]  //w
					   ));
    
    CHECK_CUDNN(cudnnSetLRNDescriptor(lrn_desc_,
				      this->size_,
				      this->alpha_,
				      this->beta_,
				      this->k_));

    this->outputs_[0]->Reshape(input_shape);
  }

  template <typename Dtype>
  void CUDNNLRNLayer<Dtype>::ForwardComputation() {
    Dtype alpha = Dtype(1);
    Dtype beta = Dtype(0);
    
    const Dtype * input_data = this->inputs_[0]->gpu_data();
    Dtype * output_data = this->outputs_[0]->mutable_gpu_data();

    CHECK_CUDNN(cudnnLRNCrossChannelForward(cudnn_handle_,
					    lrn_desc_,
					    CUDNN_LRN_CROSS_CHANNEL_DIM1,
					    &alpha,
					    input_desc_,
					    input_data,
					    &beta,
					    output_desc_,
					    output_data));
  }

  INSTANTIATE_CLASS(CUDNNLRNLayer);
} // namespace facethink
