#ifndef __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_CPU_HPP__

#include "caffe/layers/PReLU/prelu_layer_base.hpp"
#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"


namespace facethink {

  template <typename Dtype>
  class PReluLayer: public BasePReluLayer<Dtype> {
  public:
    explicit PReluLayer(bool inplace = true, bool channel_shared = true, const std::string& name="")
      :BasePReluLayer<Dtype>(inplace, channel_shared, name) {
    }

    virtual void ForwardComputation();

  private:

    DISABLE_COPY_AND_ASSIGN(PReluLayer);
  }; // class PReluLayer

} // namespace facethink

#endif
