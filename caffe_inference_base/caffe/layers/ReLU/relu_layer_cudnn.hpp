#ifndef __FACETHINK_CAFFE_LAYERS_RELU_LAYER_CUDNN_HPP__
#define __FACETHINK_CAFFE_LAYERS_RELU_LAYER_CUDNN_HPP__

#include "caffe/layers/ReLU/relu_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDNNReluLayer: public BaseReluLayer<Dtype> {
  public:
    explicit CUDNNReluLayer(bool inplace = true,
			    Dtype negative_slope = 0,
			    const std::string& name="")
      :BaseReluLayer<Dtype>(inplace, negative_slope, name),
      cudnn_handle_(nullptr) {

      InitLayer();
    }

    void InitLayer();
    ~CUDNNReluLayer();

    virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual inline std::string type() const { return "ReLU(CUDNN)"; }

  private:
    cudnnHandle_t cudnn_handle_;

    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;

    cudnnActivationDescriptor_t activ_desc_;

    DISABLE_COPY_AND_ASSIGN(CUDNNReluLayer);
  }; // class CUDNNReluLayer

} // namespace facethink


#endif
