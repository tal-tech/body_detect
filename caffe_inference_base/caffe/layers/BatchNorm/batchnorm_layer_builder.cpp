#include "caffe/layers/BatchNorm/batchnorm_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/BatchNorm/batchnorm_layer_cuda.hpp"
#else
#include "caffe/layers/BatchNorm/batchnorm_layer_cpu.hpp"
#endif



namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  BatchNormLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    bool use_global_stats;
    Dtype moving_average_fraction;
    Dtype eps;
    
    ParseLayerParameters(layer_param,
			 use_global_stats,
			 moving_average_fraction,
			 eps,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDABatchNormLayer<Dtype> >(use_global_stats,
							 moving_average_fraction,
							 eps,
							 name);
    
#else
    layer = std::make_shared<BatchNormLayer<Dtype> >(use_global_stats,
						     moving_average_fraction,
						     eps,
						     name);
#endif
    return layer;
  }

  template<typename Dtype>
  void BatchNormLayerBuilder<Dtype>::ImportData(const caffe::LayerParameter& layer_param,
						std::shared_ptr<BaseLayer<Dtype> >& layer) {
    std::shared_ptr<BaseBatchNormLayer<Dtype> > bn_layer =
      std::static_pointer_cast<BaseBatchNormLayer<Dtype> >(layer);
    
    if (IsShapeEqual(layer_param, bn_layer)) {
      std::vector<std::shared_ptr<Blob<Dtype> > > weights;
      std::shared_ptr<Blob<Dtype> > mean_weight = std::make_shared<Blob<Dtype> >();
      BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(0), mean_weight);
      weights.push_back(mean_weight);

      std::shared_ptr<Blob<Dtype> > variance_weight = std::make_shared<Blob<Dtype> >();
      BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(1), variance_weight);
      weights.push_back(variance_weight);

      std::shared_ptr<Blob<Dtype> > scale_weight = std::make_shared<Blob<Dtype> >();
      BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(2), scale_weight);
      weights.push_back(scale_weight);
      
      bn_layer->SetWeights(weights);
      
      BOOST_LOG_TRIVIAL(info)<<"BatchNormLayerBuilder: Layer("<<bn_layer->name()<<") import" 
			     <<", Mean: "<<weights[0]->shape_string()
			     <<", Variance: "<<weights[1]->shape_string()
			     <<", Scale: "<<weights[2]->shape_string();
    } else {
      BOOST_LOG_TRIVIAL(error)<< "BatchNormLayerBuilder: LayerParameter not match.";
    }
  }

  
  template <typename Dtype>
  bool BatchNormLayerBuilder<Dtype>::IsShapeEqual(const caffe::LayerParameter& layer_param,
						  std::shared_ptr<BaseBatchNormLayer<Dtype> >& layer) {
    std::string name;
    bool use_global_stats;
    Dtype moving_average_fraction;
    Dtype eps;
    
    ParseLayerParameters(layer_param,
			 use_global_stats,
			 moving_average_fraction,
			 eps,
			 name);

    return (layer->name()                    == name                     &&
	    layer->use_global_stats()        == use_global_stats         &&
	    layer->moving_average_fraction() == moving_average_fraction  &&
	    layer->eps()                     == eps );
  }
  
  template <typename Dtype>
  void BatchNormLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							  bool& use_global_stats,
							  Dtype& moving_average_fraction,
							  Dtype& eps,
							  std::string& name) {
    this->ParseLayerName(layer_param, name);
    const caffe::BatchNormParameter& param = layer_param.batch_norm_param();

    use_global_stats = true;  // Test phase

    if (param.has_moving_average_fraction()) {
      moving_average_fraction = param.moving_average_fraction();
    } else {
      moving_average_fraction = Dtype(.999);
    }

    if (param.has_eps()) {
      eps = param.eps();
    } else {
      eps = Dtype(1e-5);
    }
  }

}
