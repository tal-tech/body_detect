#ifndef __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_CPU_HPP__

#include "caffe/layers/AbsVal/absval_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class AbsValLayer: public BaseAbsValLayer<Dtype> {
  public:
    explicit AbsValLayer(const std::string& name="")
      :BaseAbsValLayer<Dtype>(name) {
    }

    virtual void ForwardComputation();

  private:

    DISABLE_COPY_AND_ASSIGN(AbsValLayer);
  }; // class AbsValLayer

}

#endif
