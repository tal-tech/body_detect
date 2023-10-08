#ifndef __FACETHINK_CAFFE_LAYERS_TANH_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_TANH_LAYER_CPU_HPP__


#include "caffe/layers/TanH/tanh_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class TanHLayer: public BaseTanHLayer<Dtype> {
  public:
    explicit TanHLayer( const std::string& name="" )
      :BaseTanHLayer<Dtype>(name) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation();

  private:

    DISABLE_COPY_AND_ASSIGN(TanHLayer);
  }; // class TanHLayer

} // namespace facethink



#endif
