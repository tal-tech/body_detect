#ifndef __FACETHINK_CAFFE_LAYERS_POOLING_LAYER_CUDNN_HPP__
#define __FACETHINK_CAFFE_LAYERS_POOLING_LAYER_CUDNN_HPP__

#include "caffe/layers/Pooling/pooling_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDNNPoolingLayer: public BasePoolingLayer<Dtype> {
  public:
    explicit CUDNNPoolingLayer(int kernel_size_h, int kernel_size_w,
			       int pad_h = 0, int pad_w = 0,
			       int stride_h = 1, int stride_w = 1,
			       PoolingMode mode = POOLING_MAX,
			       bool global_pooling = false,
			       const std::string& name = "")
      :BasePoolingLayer<Dtype>(kernel_size_h, kernel_size_w,
			       pad_h, pad_w,
			       stride_h, stride_w,
			       mode,
			       global_pooling,
			       name),
      cudnn_handle_(nullptr) {
      InitLayer();
    }

    ~CUDNNPoolingLayer();

    virtual void InitLayer();
    virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual inline std::string type() const { return "Pooling(CUDNN)"; }

  private:
    cudnnHandle_t cudnn_handle_;

    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;

    cudnnPoolingDescriptor_t  pooling_desc_;

    DISABLE_COPY_AND_ASSIGN(CUDNNPoolingLayer);
  }; // class CUDNNPoolingLayer

} // namespace facethink

#endif
