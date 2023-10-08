#ifndef __FACETHINK_CAFFE_LAYERS_INNER_PRODUCT_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_INNER_PRODUCT_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/InnerProduct/inner_product_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class InnerProductLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

    void ImportData(const caffe::LayerParameter& layer_param,
		    std::shared_ptr<BaseLayer<Dtype> >& layer);

  private:
    bool IsShapeEqual(const caffe::LayerParameter& layer_param,
		      std::shared_ptr<BaseInnerProductLayer<Dtype> >& layer);

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& num_output,
			      bool& has_bias_term,
			      int& axis,
			      bool& transpose,
			      std::string& name);

  }; // class InnerProductLayerBuilder

  REGISTER_LAYER_BUILDER(InnerProductLayerBuilder, InnerProduct);

} // namespace facethink

#endif
