#include "caffe/layers/Softmax/softmax_layer_cudnn.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDNNSoftmaxLayer<Dtype>::InitLayer() {
    if (!cudnn_handle_){
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
    }
  }

  
  template <typename Dtype>
  CUDNNSoftmaxLayer<Dtype>::~CUDNNSoftmaxLayer() {
    if (cudnn_handle_){
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
     }
  }

  template <typename Dtype>
  void CUDNNSoftmaxLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const int softmax_axis_canonical = this->inputs_[0]->CanonicalAxisIndex(this->softmax_axis_);
    int outer_num = this->inputs_[0]->count(0, softmax_axis_canonical);
    int inner_num = this->inputs_[0]->count(softmax_axis_canonical + 1);

    int N = outer_num;
    int C = this->inputs_[0]->shape(this->softmax_axis_);
    int H = inner_num;
    int W = 1;

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   N, //n
					   C, //c
					   H, //h
					   W  //w
					   ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_,
					   cudnnTensorFormat,
					   cudnnDataType,
					   N, //n
					   C, //c
					   H, //h
					   W  //w
					   ));

    this->outputs_[0]->Reshape(this->inputs_[0]->shape());
  }

  template <typename Dtype>
  void CUDNNSoftmaxLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    Dtype alpha = Dtype(1);
    Dtype beta = Dtype(0);
    
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn_handle_,
				    CUDNN_SOFTMAX_ACCURATE,
				    CUDNN_SOFTMAX_MODE_CHANNEL,
				    &alpha,
				    input_desc_,
				    input_data,
				    &beta,
				    output_desc_,
				    output_data));
  }

  INSTANTIATE_CLASS(CUDNNSoftmaxLayer);
} // namespace facethink
