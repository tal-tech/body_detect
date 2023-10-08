#ifndef __FACETHINK_CAFFE_LAYERS_SOFTMAX_LAYER_CUDNN_HPP__
#define __FACETHINK_CAFFE_LAYERS_SOFTMAX_LAYER_CUDNN_HPP__

#include "caffe/layers/Softmax/softmax_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDNNSoftmaxLayer: public BaseSoftmaxLayer<Dtype> {
  public:
    explicit CUDNNSoftmaxLayer(int softmax_axis,
			       const std::string& name = "")
      :BaseSoftmaxLayer<Dtype>(softmax_axis, name),
       cudnn_handle_(nullptr) {
      InitLayer();
    }

    ~CUDNNSoftmaxLayer();

    virtual void InitLayer();
    virtual void ForwardShape();
    virtual void ForwardComputation();


    virtual inline std::string type() const {
      return "Softmax(CUDNN)";
    }

  private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;

    DISABLE_COPY_AND_ASSIGN(CUDNNSoftmaxLayer);
  }; // class CUDNNSoftmaxLayer
} // namespace facethink


#endif
