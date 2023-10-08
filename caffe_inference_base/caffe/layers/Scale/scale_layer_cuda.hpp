#ifndef __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_CUDA_HPP__

#include "caffe/layers/Scale/scale_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAScaleLayer: public BaseScaleLayer<Dtype> {
  public:
    explicit CUDAScaleLayer(int axis = 1,
			    int num_axes = 1,
			    bool bias_term = false,
			    const std::string& name="")
      :BaseScaleLayer<Dtype>(axis, num_axes,
			     bias_term,
			     name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "Scale(CUDA)";
    }

    DISABLE_COPY_AND_ASSIGN(CUDAScaleLayer);
  }; // class CUDAScaleLayer

}  // namespace facethink

#endif
