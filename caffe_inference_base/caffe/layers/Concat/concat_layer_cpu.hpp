#ifndef __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_CPU_HPP__

#include "caffe/layers/Concat/concat_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class ConcatLayer: public BaseConcatLayer<Dtype> {
  public:
    explicit ConcatLayer(int concat_axis,
			 const std::string& name)
      :BaseConcatLayer<Dtype>(concat_axis,
			      name) {
    }

    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(ConcatLayer);
  }; // class ConcatLayer

} // namespace facethink


#endif
