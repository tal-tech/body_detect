#ifndef __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_CUDA_HPP__

#include "caffe/layers/PReLU/prelu_layer_base.hpp"
#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAPReluLayer: public BasePReluLayer<Dtype> {
  public:
    explicit CUDAPReluLayer(bool inplace = true, bool channel_shared = true, const std::string& name="")
      :BasePReluLayer<Dtype>(inplace, channel_shared, name) {
    }

    virtual void ForwardComputation();
	
	    virtual inline std::string type() const {
      return "PReLU(CUDA)";
    }

  private:

    DISABLE_COPY_AND_ASSIGN(CUDAPReluLayer);
  }; // class CUDAPReluLayer

} // namespace facethink

#endif
