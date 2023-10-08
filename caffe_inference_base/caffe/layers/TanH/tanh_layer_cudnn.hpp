#ifndef __FACETHINK_CAFFE_LAYERS_TANH_LAYER_CUDNN_HPP__
#define __FACETHINK_CAFFE_LAYERS_TANH_LAYER_CUDNN_HPP__

#include "caffe/layers/TanH/tanh_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDNNTanHLayer: public BaseTanHLayer<Dtype> {
  public:
    explicit CUDNNTanHLayer( const std::string& name="" )
      :BaseTanHLayer<Dtype>(name),
      cudnn_handle_(nullptr) {

      InitLayer();
    }

    virtual void InitLayer();
    ~CUDNNTanHLayer();

    virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual std::string type() const {
      return "TanH(CUDNN)";
    }

  private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnActivationDescriptor_t activ_desc_;

    DISABLE_COPY_AND_ASSIGN(CUDNNTanHLayer);
  }; // class CUDNNTanHLayer

} // namespace facethink

#endif
