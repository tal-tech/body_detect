#include "caffe/layers/InnerProduct/inner_product_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/InnerProduct/inner_product_layer_cuda.hpp"
#else
#include "caffe/layers/InnerProduct/inner_product_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  InnerProductLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int num_output;
    bool has_bias_term;
    int axis;
    bool transpose;
    
    ParseLayerParameters(layer_param,
			 num_output,
			 has_bias_term,
			 axis,
			 transpose,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAInnerProductLayer<Dtype> >(num_output,
							    has_bias_term,
							    axis,
							    transpose,
							    name);
#else
    layer = std::make_shared<InnerProductLayer<Dtype> >(num_output,
							has_bias_term,
							axis,
							transpose,
							name);
#endif
    return layer;
  }


  template<typename Dtype>
  void InnerProductLayerBuilder<Dtype>::ImportData(const caffe::LayerParameter& layer_param,
						   std::shared_ptr<BaseLayer<Dtype> >& layer) {
    std::shared_ptr<BaseInnerProductLayer<Dtype> > ip_layer =
      std::static_pointer_cast<BaseInnerProductLayer<Dtype> >(layer);
    
    if (IsShapeEqual(layer_param, ip_layer)) {
      std::vector<std::shared_ptr<Blob<Dtype> > > weights;
      std::shared_ptr<Blob<Dtype> > filter_weight = std::make_shared<Blob<Dtype> >();
      BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(0), filter_weight);
      weights.push_back(filter_weight);
      
      if (ip_layer->has_bias_term()) {
	std::shared_ptr<Blob<Dtype> > bias_weight = std::make_shared<Blob<Dtype> >();
	BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(1), bias_weight);
	weights.push_back(bias_weight);
      }

      ip_layer->SetWeights(weights);
      
      BOOST_LOG_TRIVIAL(info)<<"InnerProductLayerBuilder: Layer("<<ip_layer->name()<<") import" 
			     <<", Weights: "<<weights[0]->shape_string()
			     <<", Bias: "<< (ip_layer->has_bias_term()? weights[1]->shape_string():"NULL");
    } else {
      BOOST_LOG_TRIVIAL(error)<< "InnerProductLayerBuilder: LayerParameter not match.";
    }
  }
  
  template <typename Dtype>
  bool InnerProductLayerBuilder<Dtype>::IsShapeEqual(const caffe::LayerParameter& layer_param,
						     std::shared_ptr<BaseInnerProductLayer<Dtype> >& layer) {
    std::string name;
    int num_output;
    bool has_bias_term;
    int axis;
    bool transpose;

    ParseLayerParameters(layer_param,
			 num_output,
			 has_bias_term,
			 axis,
			 transpose,
			 name);

    return (layer->name()          == name          &&
	    layer->has_bias_term() == has_bias_term &&
	    layer->num_output()    == num_output    &&
	    layer->axis()          == axis          &&
	    layer->transpose()     == transpose );
  }

  template <typename Dtype>
  void InnerProductLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							     int& num_output,
							     bool& has_bias_term,
							     int& axis,
							     bool& transpose,
							     std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::InnerProductParameter& ip_param = layer_param.inner_product_param();

    if (ip_param.has_num_output()){
      num_output = ip_param.num_output();
    }else{
      BOOST_LOG_TRIVIAL(error)<<"InnerProductLayerBuilder: num_output no found.";
    }

    if (ip_param.has_bias_term()){
      has_bias_term = ip_param.bias_term();
    }else{
      has_bias_term = true;
    }

    if (ip_param.has_axis()){
      axis = ip_param.axis();
    }else{
      axis = 1;
    }

    if (ip_param.has_transpose()){
      transpose = ip_param.transpose();
    }else{
      transpose = false;
    }
  }

} // namespace facethink
