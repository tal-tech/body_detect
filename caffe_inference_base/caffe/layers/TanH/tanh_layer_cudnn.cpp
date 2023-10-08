#include "caffe/layers/TanH/tanh_layer_cudnn.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDNNTanHLayer<Dtype>::InitLayer() {
    if (!cudnn_handle_) {
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
      CHECK_CUDNN(cudnnCreateActivationDescriptor(&activ_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
    }
  }

  template <typename Dtype>
  CUDNNTanHLayer<Dtype>::~CUDNNTanHLayer() {
    if (cudnn_handle_){
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
      CHECK_CUDNN(cudnnDestroyActivationDescriptor(activ_desc_));
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
    }
  }

  template <typename Dtype>
  void CUDNNTanHLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int> input_shape = this->inputs_[0]->shape();
    std::vector<int> magic_input_shape = {1, 1, 1, 1};

    for (size_t i=0; i<input_shape.size(); ++i){
      magic_input_shape[i] = input_shape[i];
    }
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   magic_input_shape[0], //n
					   magic_input_shape[1], //c
					   magic_input_shape[2], //h
					   magic_input_shape[3]  //w
					   ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   magic_input_shape[0], //n
					   magic_input_shape[1], //c
					   magic_input_shape[2], //h
					   magic_input_shape[3]  //w
					   ));

    CHECK_CUDNN(cudnnSetActivationDescriptor(activ_desc_,
					     CUDNN_ACTIVATION_TANH,
					     CUDNN_PROPAGATE_NAN,
					     Dtype(0)
					     ));
  

    this->outputs_[0]->Reshape(input_shape);
  }


  template <typename Dtype>
  void CUDNNTanHLayer<Dtype>::ForwardComputation() {
    Dtype alpha = Dtype(1);
    Dtype beta = Dtype(0);

    const Dtype * input_data = this->inputs_[0]->gpu_data();
    Dtype * output_data = this->outputs_[0]->mutable_gpu_data();
    
    CHECK_CUDNN(cudnnActivationForward(cudnn_handle_,
				       activ_desc_,
				       &alpha,
				       input_desc_,
				       input_data,
				       &beta,
				       output_desc_,
				       output_data
				       ));	
  }

  INSTANTIATE_CLASS(CUDNNTanHLayer);
} // namespace facethink
