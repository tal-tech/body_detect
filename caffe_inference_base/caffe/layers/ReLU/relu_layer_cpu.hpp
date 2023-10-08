#ifndef __FACETHINK_CAFFE_LAYERS_RELU_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_RELU_LAYER_CPU_HPP__

#include "caffe/layers/ReLU/relu_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class ReluLayer: public BaseReluLayer<Dtype> {
  public:
    explicit ReluLayer(bool inplace = true,
		       Dtype negative_slope = 0,
		       const std::string& name="")
      :BaseReluLayer<Dtype>(inplace, negative_slope, name) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation();

  private:

    DISABLE_COPY_AND_ASSIGN(ReluLayer);
  }; // class ReluLayer

} // namespace facethink

#endif
