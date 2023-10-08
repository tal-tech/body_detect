#ifndef __FACETHINK_CAFFE_LAYERS_BATCHNORM_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_BATCHNORM_LAYER_CPU_HPP__

#include "caffe/layers/BatchNorm/batchnorm_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class BatchNormLayer: public BaseBatchNormLayer<Dtype> {
  public:
    explicit BatchNormLayer(bool use_global_stats,
			    Dtype moving_average_fraction = Dtype(.999),
			    Dtype eps = Dtype(1e-5),
			    const std::string& name="")
      :BaseBatchNormLayer<Dtype>(use_global_stats,
				 moving_average_fraction,
				 eps,
				 name) {
    }

    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(BatchNormLayer);
  }; // class BatchNormLayer

} // namespace facethink



#endif
