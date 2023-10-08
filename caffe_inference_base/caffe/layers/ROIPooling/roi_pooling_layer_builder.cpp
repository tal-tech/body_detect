#include "caffe/layers/ROIPooling/roi_pooling_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/ROIPooling/roi_pooling_layer_cuda.hpp"
#else
#include "caffe/layers/ROIPooling/roi_pooling_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ROIPoolingLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int pooled_h, pooled_w;
    Dtype spatial_scale;
    
    ParseLayerParameters(layer_param, pooled_h, pooled_w, spatial_scale, name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAROIPoolingLayer<Dtype> >(pooled_h,
							  pooled_w,
							  spatial_scale,
							  name);
   
#else
    layer = std::make_shared<ROIPoolingLayer<Dtype> >(pooled_h,
						      pooled_w,
						      spatial_scale,
						      name);
#endif
    return layer;
  }

  template <typename Dtype>
  void ROIPoolingLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							   int& pooled_h,
							   int& pooled_w,
							   Dtype& spatial_scale,
							   std::string& name) {
    this->ParseLayerName(layer_param, name);
    
    const caffe::ROIPoolingParameter& roi_param = layer_param.roi_pooling_param();
    
    if (roi_param.has_pooled_h()){
      pooled_h = roi_param.pooled_h();
    }else{
      pooled_h = 0;
    }

    if (roi_param.has_pooled_w()){
      pooled_w = roi_param.pooled_w();
    }else{
      pooled_w = 0;
    }

    if (roi_param.has_spatial_scale()){
      spatial_scale = roi_param.spatial_scale();
    }else{
      spatial_scale = 1;
    }
    
  }

} // namespace facethink

