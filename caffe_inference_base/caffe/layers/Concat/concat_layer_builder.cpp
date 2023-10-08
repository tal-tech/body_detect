#include "caffe/layers/Concat/concat_layer_builder.hpp"


#ifndef CPU_ONLY
#include "caffe/layers/Concat/concat_layer_cuda.hpp"
#else
#include "caffe/layers/Concat/concat_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ConcatLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int concat_axis;
    
    ParseLayerParameters(layer_param,
			 concat_axis,
			 name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAConcatLayer<Dtype> >(concat_axis, name);
#else			
    layer = std::make_shared<ConcatLayer<Dtype> >(concat_axis, name);
#endif
    return layer;
  }

  template <typename Dtype>
  void ConcatLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						       int& concat_axis,
						       std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::ConcatParameter& concat_param = layer_param.concat_param();
    if (concat_param.has_concat_dim()){
      concat_axis = static_cast<int>(concat_param.concat_dim());
      // Don't allow negative indexing for concat_dim, a uint32 -- almost
      // certainly unintended.
      if (concat_axis < 0)
	BOOST_LOG_TRIVIAL(error)<< "ConcatLayerBuilder: casting concat_dim from uint32 to int32 "
				<< "produced negative result; concat_dim must satisfy "
				<< "0 <= concat_dim";
  
    }else if(concat_param.has_axis()){
      concat_axis = concat_param.axis();
    }else{
      concat_axis = 1;
    }
  }
  
} // namespace facethink;
