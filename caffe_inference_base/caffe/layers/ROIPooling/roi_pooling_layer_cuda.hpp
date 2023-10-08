#ifndef __FACETHINK_CAFFE_LAYERS_ROI_POOLING_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_ROI_POOLING_LAYER_CUDA_HPP__

#include "caffe/layers/ROIPooling/roi_pooling_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAROIPoolingLayer: public BaseROIPoolingLayer<Dtype> {
  public:
    explicit CUDAROIPoolingLayer(int pooled_h,
				 int pooled_w,
				 Dtype spatial_scale,
				 const std::string& name)
      :BaseROIPoolingLayer<Dtype>(pooled_h,
				  pooled_w,
				  spatial_scale,
				  name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const{
      return "ROIPooling(CUDA)";
    }

    DISABLE_COPY_AND_ASSIGN(CUDAROIPoolingLayer);
  };

} // namespace facethink

#endif
