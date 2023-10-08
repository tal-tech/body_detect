#ifndef __FACETHINK_CAFFE_NETBUILDER_POOLING_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_NETBUILDER_POOLING_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/Pooling/pooling_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class PoolingLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& kernel_size_h, int& kernel_size_w,
			      int& pad_h, int& pad_w,
			      int& stride_h, int& stride_w,
			      PoolingMode& mode,
			      bool& global_pooling,
			      std::string& name);

  }; // class PoolingLayerBuilder

  REGISTER_LAYER_BUILDER(PoolingLayerBuilder, Pooling);
} // namespace facethink


#endif
