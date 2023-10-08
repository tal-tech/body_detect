#ifndef __FACETHINK_CAFFE_LAYERS_LRN_LAYER_CUDNN_HPP__
#define __FACETHINK_CAFFE_LAYERS_LRN_LAYER_CUDNN_HPP__

#include "caffe/layers/LRN/lrn_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDNNLRNLayer: public BaseLRNLayer<Dtype> {
  public:
    explicit CUDNNLRNLayer(int size,
			   Dtype alpha,
			   Dtype beta,
			   Dtype k,
			   const std::string& name="")
      :BaseLRNLayer<Dtype> (size, alpha, beta, k, name),
      cudnn_handle_(nullptr) {

      InitLayer();
    }

    virtual void InitLayer();
    ~CUDNNLRNLayer();

    virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "LRN(CUDNN)";
    }

  private:
    cudnnHandle_t cudnn_handle_;

    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;

    cudnnLRNDescriptor_t lrn_desc_;
  }; // class CUDNNLRNLayer

} // namespace facethink

#endif
