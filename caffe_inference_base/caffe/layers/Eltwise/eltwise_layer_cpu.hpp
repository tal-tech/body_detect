#ifndef __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_CPU_HPP__

#include "caffe/layers/Eltwise/eltwise_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class EltwiseLayer: public BaseEltwiseLayer<Dtype> {
  public:
    explicit EltwiseLayer(EltwiseOp op,
			  const std::vector<Dtype> coeffs,
			  const std::string& name="")
      :BaseEltwiseLayer<Dtype>(op,
			       coeffs,
			       name) {

    }

    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(EltwiseLayer);
  }; // class EltwiseLayer

} // namespace facethink


#endif
