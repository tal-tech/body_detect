#ifndef __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_CPU_HPP__

#include "caffe/layers/Normalize/normalize_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class NormalizeLayer: public BaseNormalizeLayer<Dtype> {
  public:
    explicit NormalizeLayer(bool across_spatial = true,
			    bool channel_shared = true,
			    Dtype eps = Dtype(1e-10),
			    const std::string& name="")
      :BaseNormalizeLayer<Dtype>(across_spatial,
				 channel_shared,
				 eps,
				 name) {
    }

    virtual void ForwardComputation();

  private:

    DISABLE_COPY_AND_ASSIGN(NormalizeLayer);
  }; //class CUDANormalizeLayer

} // namespace facethink


#endif
