#ifndef __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_CUDA_HPP__

#include "caffe/layers/Permute/permute_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAPermuteLayer: public BasePermuteLayer<Dtype> {
  public:
    explicit CUDAPermuteLayer(const std::vector<int>& permute_order,
			      const std::string& name="")
      :BasePermuteLayer<Dtype>(permute_order, name) {

    }

    virtual void ForwardComputation();

    virtual inline std::string type() const { return "Permute(CUDA)"; }

    DISABLE_COPY_AND_ASSIGN(CUDAPermuteLayer);
  }; // class CUDAPermuteLayer

} // namespace facethink


#endif
