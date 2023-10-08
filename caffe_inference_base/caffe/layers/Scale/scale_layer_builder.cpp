#include "caffe/layers/Scale/scale_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Scale/scale_layer_cuda.hpp"
#else
#include "caffe/layers/Scale/scale_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ScaleLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int axis;
    int num_axes;
    bool bias_term;
    
    ParseLayerParameters(layer_param,
			 axis,
			 num_axes,
			 bias_term,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAScaleLayer<Dtype> >(axis,
						     num_axes,
						     bias_term,
						     name);
    
#else
    layer = std::make_shared<ScaleLayer<Dtype> >(axis,
						 num_axes,
						 bias_term,
						 name);
#endif
    return layer;
  }

  template<typename Dtype>
  void ScaleLayerBuilder<Dtype>::ImportData(const caffe::LayerParameter& layer_param,
					    std::shared_ptr<BaseLayer<Dtype> >& layer) {
    std::shared_ptr<BaseScaleLayer<Dtype> > scale_layer =
      std::static_pointer_cast<BaseScaleLayer<Dtype> >(layer);
    
    if (IsShapeEqual(layer_param, scale_layer)) {
      std::vector<std::shared_ptr<Blob<Dtype> > > weights;

      const int blob_size = layer_param.blobs_size();
      for (int i = 0; i < blob_size; ++i) {
	std::shared_ptr<Blob<Dtype> > weight = std::make_shared<Blob<Dtype> >();
	BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(i), weight);
	weights.push_back(weight);
      }
      
      scale_layer->SetWeights(weights);

      if (weights.size() == 1) {
	BOOST_LOG_TRIVIAL(info)<<"ScaleLayerBuilder: Layer("<<scale_layer->name()<<") import" 
			       <<" bias: "<<weights[0]->shape_string();
      } else {
	BOOST_LOG_TRIVIAL(info)<<"ScaleLayerBuilder: Layer("<<scale_layer->name()<<") import" 
			       <<" scale: "<<weights[0]->shape_string()
			       <<", bias: "<<weights[1]->shape_string();
      }
      
    } else {
      BOOST_LOG_TRIVIAL(error)<< "ScaleLayerBuilder: LayerParameter not match.";
    }
  }
  

  template <typename Dtype>
  bool ScaleLayerBuilder<Dtype>::IsShapeEqual(const caffe::LayerParameter& layer_param,
					      std::shared_ptr<BaseScaleLayer<Dtype> >& layer) {
    std::string name;
    int axis;
    int num_axes;
    bool bias_term;
    
        
    ParseLayerParameters(layer_param,
			 axis,
			 num_axes,
			 bias_term,
			 name);
    
    return (layer->name()        == name      &&
	    layer->axis()        == axis      &&
	    layer->num_axes()    == num_axes  &&
	    layer->bias_term()   == bias_term );
  }
  

  template <typename Dtype>
  void ScaleLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						      int& axis,
						      int& num_axes,
						      bool& bias_term,
						      std::string& name) {
    this->ParseLayerName(layer_param, name);
    
    const caffe::ScaleParameter& scale_param = layer_param.scale_param();

    if (scale_param.has_axis()) {
      axis = scale_param.axis();
    } else {
      axis = 1;
    }

    if (scale_param.has_num_axes()) {
      num_axes = scale_param.num_axes();
    } else {
      num_axes = 1;
    }

    if (scale_param.has_bias_term()) {
      bias_term = scale_param.bias_term();
    } else {
      bias_term = false;
    }
  }


} // namespace facethink
