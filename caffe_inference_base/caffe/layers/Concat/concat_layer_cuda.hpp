#ifndef __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_CUDA_HPP__

#include "caffe/layers/Concat/concat_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAConcatLayer: public BaseConcatLayer<Dtype> {
  public:
    explicit CUDAConcatLayer(int concat_axis,
			     const std::string& name)
      :BaseConcatLayer<Dtype>(concat_axis,
			      name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "Concat(CUDA)";
    }

    DISABLE_COPY_AND_ASSIGN(CUDAConcatLayer);
  }; // class CUDAConcatLayer

} // namespace facethink

#endif
