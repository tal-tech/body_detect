#include "caffe/layers/Reshape/reshape_layer_builder.hpp"

#include "caffe/layers/Reshape/reshape_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ReshapeLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    std::vector<int> dims;
    int axis;
    int num_axes;
    
    ParseLayerParameters(layer_param,
			 dims,
			 axis,
			 num_axes,
			 name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;

    layer = std::make_shared<BaseReshapeLayer<Dtype> >(dims,
						       axis,
						       num_axes,
						       name);

    return layer;
  }

  template <typename Dtype>
  void ReshapeLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							std::vector<int>& dims,
							int& axis,
							int& num_axes,
							std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::BlobShape& top_blob_shape = layer_param.reshape_param().shape();
    const int top_num_axes = top_blob_shape.dim_size();

    dims.clear();
    for (int i = 0; i < top_num_axes; ++i) {
      dims.push_back(top_blob_shape.dim(i));
    }

    if (layer_param.reshape_param().has_axis()) {
      axis = layer_param.reshape_param().axis();
    }else{
      axis = 0;
    }

    if (layer_param.reshape_param().has_num_axes()) {
      num_axes = layer_param.reshape_param().num_axes();
    }else{
      num_axes = -1;
    }
  }

  
} // namespace facethink
