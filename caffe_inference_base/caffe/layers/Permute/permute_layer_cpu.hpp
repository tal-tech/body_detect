#ifndef __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_CPU_HPP__

#include "caffe/layers/Permute/permute_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class PermuteLayer: public BasePermuteLayer<Dtype> {
  public:
    explicit PermuteLayer(const std::vector<int>& permute_order,
			      const std::string& name="")
      :BasePermuteLayer<Dtype>(permute_order, name) {
    }

    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(PermuteLayer);
  }; // class PermuteLayer

} // namespace facethink



#endif
