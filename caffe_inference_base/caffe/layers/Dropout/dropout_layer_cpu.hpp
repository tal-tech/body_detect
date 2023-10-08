#ifndef __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_CPU_HPP__

#include "caffe/layers/Dropout/dropout_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class DropoutLayer: public BaseDropoutLayer<Dtype> {
  public:
    explicit DropoutLayer(Dtype dropout_ratio,
			  const std::string& name="")
      :BaseDropoutLayer<Dtype>(dropout_ratio, name) {
    }

    virtual void ForwardComputation();
  }; // DropoutLayer

}



#endif
