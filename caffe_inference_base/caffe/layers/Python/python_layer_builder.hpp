#ifndef __FACETHINK_CAFFE_NETBUILDER_PYTHON_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_NETBUILDER_PYTHON_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class PythonLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:
    std::shared_ptr<BaseLayer<Dtype> > CreateProposalLayer(const caffe::LayerParameter& layer_param);
    void ParseProposalLayerParameters(const caffe::LayerParameter& layer_param,
				      int& feat_stride,
				      std::vector<Dtype>& scales,
				      int& allowed_border,
				      std::string& name);

  }; // class PythonLayerBuilder

  REGISTER_LAYER_BUILDER(PythonLayerBuilder, Python);

} // namespace facethink

#endif
