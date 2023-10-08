#ifndef __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_CPU_HPP__

#include "caffe/layers/Scale/scale_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class ScaleLayer: public BaseScaleLayer<Dtype> {
  public:
    explicit ScaleLayer(int axis = 1,
			int num_axes = 1,
			bool bias_term = false,
			const std::string& name="")
      :BaseScaleLayer<Dtype>(axis, num_axes,
			     bias_term,
			     name) {
    }

    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(ScaleLayer);
  }; // class ScaleLayer


} // namespace facethink

#endif
