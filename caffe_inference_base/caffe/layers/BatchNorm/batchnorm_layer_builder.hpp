#ifndef __FACETHINK_CAFFE_LAYERS_BATCHNORM_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_BATCHNORM_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/BatchNorm/batchnorm_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class BatchNormLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

    void ImportData(const caffe::LayerParameter& layer_param,
		    std::shared_ptr<BaseLayer<Dtype> >& layer);

  private:
    bool IsShapeEqual(const caffe::LayerParameter& layer_param,
		      std::shared_ptr<BaseBatchNormLayer<Dtype> >& layer);

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      bool& use_global_stats,
			      Dtype& moving_average_fraction,
			      Dtype& eps,
			      std::string& name);

  }; // class BatchNormLayerBuilder

  REGISTER_LAYER_BUILDER(BatchNormLayerBuilder, BatchNorm);

} // namespace facethink

#endif
