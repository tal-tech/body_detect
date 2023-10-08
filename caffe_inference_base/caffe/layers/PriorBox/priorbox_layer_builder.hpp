#ifndef __FACETHINK_CAFFE_LAYERS_PRIORBOX_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRIORBOX_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class PriorBoxLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      Dtype& min_size,
			      Dtype& max_size,
			      std::vector<Dtype>& aspect_ratios,
			      std::vector<Dtype>& variances,
			      bool& flip,
			      bool& clip,
			      int& img_h, int& img_w,
			      Dtype& step_h, Dtype& step_w,
			      Dtype& offset,
			      std::string& name);

  }; // class PriorBoxLayerBuilder

  REGISTER_LAYER_BUILDER(PriorBoxLayerBuilder, PriorBox);

} // namespace facethink

#endif
