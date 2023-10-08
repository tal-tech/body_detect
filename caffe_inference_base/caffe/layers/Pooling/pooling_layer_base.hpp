#ifndef __FACETHINK_CAFFE_LAYERS_POOLING_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_POOLING_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  enum PoolingMode { POOLING_MAX, POOLING_AVE };
  
  template <typename Dtype>
  class BasePoolingLayer: public BaseLayer<Dtype> {
  public: 
    explicit BasePoolingLayer(int kernel_size_h, int kernel_size_w,
			      int pad_h = 0, int pad_w = 0,
			      int stride_h = 1, int stride_w = 1,
			      PoolingMode mode = POOLING_MAX,
			      bool global_pooling = false,
			      const std::string& name = "")
      :BaseLayer<Dtype>(name),
      kernel_size_h_(kernel_size_h), kernel_size_w_(kernel_size_w),
      pad_h_(pad_h), pad_w_(pad_w),
      stride_h_(stride_h), stride_w_(stride_w),
      mode_(mode), global_pooling_(global_pooling) {
    }

    virtual void ForwardShape() = 0;
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const { return "Pooling"; }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", mode: "<< (mode_ == POOLING_MAX ? "MAX":"AVE")
	     <<", kernel size: ("<<kernel_size_h_<<","<<kernel_size_w_<<")"
	     <<", pad: ("<<pad_h_<<","<<pad_w_<<")"
	     <<", stride: ("<<stride_h_<<","<<stride_w_<<")"
	     <<", global_pooling: " << (global_pooling_? "True":"False");
      return stream.str();
    }

    /// set and output parameters
    inline int kernel_size_h() const { return kernel_size_h_; }
    inline int kernel_size_w() const { return kernel_size_w_; }
    inline int pad_h() const { return pad_h_; }
    inline int pad_w() const { return pad_w_; }
    inline int stride_h() const { return stride_h_; }
    inline int stride_w() const { return stride_w_; }
    inline bool global_pooling() const { return global_pooling_; }
    inline PoolingMode mode() const { return mode_; }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"PoolingLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

    virtual inline bool CheckMode() const {
      if (mode_ != POOLING_MAX && mode_ != POOLING_AVE) {
	BOOST_LOG_TRIVIAL(error)<<"PoolingLayer: Unsupported pooling method (must be POOLING_MAX or POOLING_AVE)";
	return false;
      }
      return true;
    }

    void ComputeOutputShape(const std::vector<int>& input_shape,
			    std::vector<int>& output_shape);
    
  protected:
    PoolingMode mode_;
    int kernel_size_h_, kernel_size_w_;
    int pad_h_, pad_w_;
    int stride_h_, stride_w_;
    bool global_pooling_;

    DISABLE_COPY_AND_ASSIGN(BasePoolingLayer);
  }; // class BasePoolingLayer

} // namespace facethink

#endif
