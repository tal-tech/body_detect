#ifndef __FACETHINK_CAFFE_LAYER_POOLING_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYER_POOLING_LAYER_CPU_HPP__

#include "caffe/layers/Pooling/pooling_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class PoolingLayer: public BasePoolingLayer<Dtype> {
  public:
    explicit PoolingLayer(int kernel_size_h, int kernel_size_w,
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
			       name) {

    }

    virtual void ForwardShape();
    virtual void ForwardComputation();

  private:

    DISABLE_COPY_AND_ASSIGN(PoolingLayer);
  }; // class PoolingLayer

} // namespace facethink

#endif
