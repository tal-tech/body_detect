#ifndef __FACETHINK_CAFFE_LAYERS_ROI_POOLING_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_ROI_POOLING_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseROIPoolingLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseROIPoolingLayer(int pooled_h,
				 int pooled_w,
				 Dtype spatial_scale,
				 const std::string& name)
      :BaseLayer<Dtype>(name),
      pooled_h_(pooled_h),
      pooled_w_(pooled_w),
      spatial_scale_(spatial_scale) {

    }

    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual inline std::string type() const{
      return "ROIPooling";
    }

    virtual inline std::string param_string() const{
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", pooled_size: ("<<pooled_h_<<", "<<pooled_w_<<")"
	    <<", spatial_scale: "<<spatial_scale_;	
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 2 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"ROIPoolingLayer: only accept two input blobs and one output blob.";
	return false;
      }
      return true;
    } 

  protected:
    int pooled_h_, pooled_w_;
    Dtype spatial_scale_;
    
    Blob<int>  max_idx_;

    DISABLE_COPY_AND_ASSIGN(BaseROIPoolingLayer);
    
  }; // class ROIPoolingLayer
} // namespace facethink

#endif
