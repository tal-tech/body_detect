#ifndef __FACETHINK_CAFFE_CORE_BASE_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_CORE_BASE_LAYER_BUILDER_HPP__

#include "caffe/core/common.hpp"
#include "caffe/core/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace facethink {

  template<typename Dtype>
  class BaseLayerBuilder {
  public:
    virtual std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param) = 0;
    
    virtual void ImportData(const caffe::LayerParameter& layer_param,
			    std::shared_ptr<BaseLayer<Dtype> >& layer) {
      BOOST_LOG_TRIVIAL(error)<<"LayerBuilder: this "<<layer->type()<<"Layer "
			      <<layer->name()<<" shouldn't import data.";
    }

  protected:
    virtual inline void ParseLayerName(const caffe::LayerParameter& layer_param,
				       std::string& name) {
      name = "";
      if (layer_param.has_name()){
	name = layer_param.name();
      }
    }
    
  }; // class BaseLayerBuilder

} // namespace facethink

#endif
