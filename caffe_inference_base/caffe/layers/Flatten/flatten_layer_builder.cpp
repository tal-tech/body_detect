#include "caffe/layers/Flatten/flatten_layer_builder.hpp"

#include "caffe/layers/Flatten/flatten_layer_base.hpp"


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  FlattenLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int start_axis;
    int end_axis;

    ParseLayerParameters(layer_param,
			 start_axis,
			 end_axis,
			 name);


    std::shared_ptr<BaseLayer<Dtype> > layer
      = std::make_shared<BaseFlattenLayer<Dtype> >(start_axis,
						   end_axis,
						   name);

    return layer;
  }


  template <typename Dtype>
  void FlattenLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							int& start_axis,
							int& end_axis,
							std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::FlattenParameter& flatten_param = layer_param.flatten_param();

    if (flatten_param.has_axis()) {
      start_axis = flatten_param.axis();
    }else{
      start_axis = 1;
    }

    if (flatten_param.has_end_axis()) {
      end_axis = flatten_param.end_axis();
    }else{
      end_axis = -1;
    }
  }
  
} // namespace facethink
