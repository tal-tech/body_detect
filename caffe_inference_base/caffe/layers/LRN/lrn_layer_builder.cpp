#include "caffe/layers/LRN/lrn_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/LRN/lrn_layer_cudnn.hpp"
#else
#include "caffe/layers/LRN/lrn_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  LRNLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int size;
    Dtype alpha, beta, k;
    
    ParseLayerParameters(layer_param,
			 size,
			 alpha, beta, k,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDNNLRNLayer<Dtype> >(size,
						    alpha, beta, k,
						    name);
   
#else
    layer = std::make_shared<LRNLayer<Dtype> >(size,
    					       alpha, beta, k,
     					       name);
#endif
    return layer;
  }

  template <typename Dtype>
  void LRNLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						    int& size,
						    Dtype& alpha,
						    Dtype& beta,
						    Dtype& k,
						    std::string& name) {
    this->ParseLayerName(layer_param, name);
    
    const caffe::LRNParameter& lrn_param = layer_param.lrn_param();

    if (lrn_param.has_local_size()){
      size = lrn_param.local_size();
    }else{
      size = 5;
    }

    if (lrn_param.has_alpha()){
      alpha = lrn_param.alpha();
    }else{
      alpha = Dtype(1.0);
    }

    if (lrn_param.has_beta()){
      beta = lrn_param.beta();
    }else{
      beta = Dtype(0.75);
    }

    if (lrn_param.has_k()){
      k = lrn_param.k();
    }else{
      k = Dtype(1.0);
    }

    if (lrn_param.norm_region() != caffe::LRNParameter_NormRegion_ACROSS_CHANNELS) {
      BOOST_LOG_TRIVIAL(error)<<"LRNLayerBuilder: find unsupported NormRegion method.";
    }
  }
 
}
