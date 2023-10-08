#ifndef __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/Scale/scale_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class ScaleLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

    void ImportData(const caffe::LayerParameter& layer_param,
		    std::shared_ptr<BaseLayer<Dtype> >& layer);

  private:
    bool IsShapeEqual(const caffe::LayerParameter& layer_param,
		      std::shared_ptr<BaseScaleLayer<Dtype> >& layer);

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& axis,
			      int& num_axes,
			      bool& bias_term,
			      std::string& name);

  }; // class BatchNormLayerBuilder

  REGISTER_LAYER_BUILDER(ScaleLayerBuilder, Scale);

} // namespace facethink


#endif
