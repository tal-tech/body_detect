#ifndef __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/Normalize/normalize_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class NormalizeLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

    void ImportData(const caffe::LayerParameter& layer_param,
		    std::shared_ptr<BaseLayer<Dtype> >& layer);

  private:
    bool IsShapeEqual(const caffe::LayerParameter& layer_param,
		      std::shared_ptr<BaseNormalizeLayer<Dtype> >& layer);

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      bool& across_spatial,
			      bool& channel_shared,
			      Dtype& eps,
			      std::string& name);
  }; // class NormalizeLayerBuilder

  REGISTER_LAYER_BUILDER(NormalizeLayerBuilder, Normalize);

} // namespace facethink


#endif
