#include "caffe/layers/Normalize/normalize_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Normalize/normalize_layer_cuda.hpp"
#else
#include "caffe/layers/Normalize/normalize_layer_cpu.hpp"
#endif

namespace facethink {
  
  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  NormalizeLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    bool across_spatial;
    bool channel_shared;
    Dtype eps;
    ParseLayerParameters(layer_param,
			 across_spatial,
			 channel_shared,
			 eps,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDANormalizeLayer<Dtype> >(across_spatial,
							 channel_shared,
							 eps,
							 name);
   
#else
    layer = std::make_shared<NormalizeLayer<Dtype> >(across_spatial,
						     channel_shared,
						     eps,
						     name);
#endif
    return layer;
  }

  template<typename Dtype>
  void NormalizeLayerBuilder<Dtype>::ImportData(const caffe::LayerParameter& layer_param,
						std::shared_ptr<BaseLayer<Dtype> >& layer) {
    std::shared_ptr<BaseNormalizeLayer<Dtype> > norm_layer =
      std::static_pointer_cast<BaseNormalizeLayer<Dtype> >(layer);
    
    if (IsShapeEqual(layer_param, norm_layer)) {
      std::vector<std::shared_ptr<Blob<Dtype> > > weights;
      std::shared_ptr<Blob<Dtype> > scale_weight = std::make_shared<Blob<Dtype> >();
      BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(0), scale_weight);
      weights.push_back(scale_weight);
    
      norm_layer->SetWeights(weights);
      
      BOOST_LOG_TRIVIAL(info)<<"NormalizeLayerBuilder: Layer("<<norm_layer->name()<<") import" 
			     <<" Weights: "<<weights[0]->shape_string();
    } else {
      BOOST_LOG_TRIVIAL(error)<<"NormalizeLayerBuilder: LayerParameter not match.";
    }
  }

  template <typename Dtype>
  bool NormalizeLayerBuilder<Dtype>::IsShapeEqual(const caffe::LayerParameter& layer_param,
						  std::shared_ptr<BaseNormalizeLayer<Dtype> >& layer) {
    std::string name;
    bool across_spatial;
    bool channel_shared;
    Dtype eps;

    ParseLayerParameters(layer_param,
		         across_spatial,
			 channel_shared,
			 eps,
			 name);

    return (layer->name()           == name               &&
	    layer->across_spatial() == across_spatial     &&
	    layer->channel_shared() == channel_shared     &&
	    abs(layer->eps() - eps) < 1e-9 );
  }

  
  template <typename Dtype>
  void NormalizeLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							  bool& across_spatial,
							  bool& channel_shared,
							  Dtype& eps,
							  std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::NormalizeParameter& norm_param = layer_param.norm_param();

    if (norm_param.has_across_spatial()){
      across_spatial = norm_param.across_spatial();
    }else{
      across_spatial = true;
    }

    if (norm_param.has_channel_shared()){
      channel_shared = norm_param.channel_shared();
    }else{
      channel_shared = true;
    }

    if (norm_param.has_eps()){
      eps = static_cast<Dtype>(norm_param.eps());
    }else{
      eps = static_cast<Dtype>(1e-10);
    }
  }

} // namespace facethink


