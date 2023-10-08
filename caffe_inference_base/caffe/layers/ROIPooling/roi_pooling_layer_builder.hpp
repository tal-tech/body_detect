#ifndef __FACETHINK_CAFFE_NETBUILDER_ROI_POOLING_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_NETBUILDER_ROI_POOLING_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class ROIPoolingLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& pooled_h,
			      int& pooled_w,
			      Dtype& spatial_scale,
			      std::string& name);

  }; // class ROIPoolingLayerBuilder

  REGISTER_LAYER_BUILDER(ROIPoolingLayerBuilder, ROIPooling);

} // namespace facethink

#endif
