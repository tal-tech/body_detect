#ifndef __FACETHINK_CAFFE_LAYERS_CONV_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONV_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseConvLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseConvLayer(int num_output,
			   int kernel_size_h, int kernel_size_w,
			   int pad_h = 0, int pad_w = 0,
			   int stride_h = 1, int stride_w = 1,
			   int dilation_h = 1, int dilation_w = 1,
			   bool has_bias_term = true,
			   int group = 1,
			   const std::string& name = "")
      :BaseLayer<Dtype>(name),
      num_output_(num_output),
      kernel_size_h_(kernel_size_h), kernel_size_w_(kernel_size_w),
      pad_h_(pad_h), pad_w_(pad_w),
      stride_h_(stride_h), stride_w_(stride_w),
      dilation_h_(dilation_h), dilation_w_(dilation_w),
      has_bias_term_(has_bias_term),
      group_(group) {
    }
    
    virtual void ForwardShape() = 0;
    virtual void ForwardComputation() = 0;
   
    // output convolution weights and bias
    virtual inline const std::shared_ptr<Blob<Dtype> > weight() {
      if (this->weights_.empty()){
	BOOST_LOG_TRIVIAL(error)<< "ConvlutionLayer: No weight found.";
	return nullptr;
      }
      return this->weights_[0];
    }
    virtual inline const std::shared_ptr<Blob<Dtype> >  bias() {
      if (this->weights_.size()<2 || !has_bias_term_) {
	BOOST_LOG_TRIVIAL(error)<< "ConvlutionLayer: No bias found.";
	return nullptr;
      }
      return this->weights_[1];
    }

    virtual inline std::string type() const {
      return "Convolution";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", num_output: "<<num_output_
	     <<", kernel size: ("<<kernel_size_h_<<","<<kernel_size_w_<<")"
	     <<", pad: ("<<pad_h_<<","<<pad_w_<<")"
	     <<", stride: ("<<stride_h_<<","<<stride_w_<<")"
	     <<", dilation: ("<<dilation_h_<<","<<dilation_w_<<")"
	     <<", group: "<<group_
	     <<", bias: "<<( has_bias_term_? "True":"False" );
      return stream.str();
    }

    /// set and output parameters
    virtual inline bool has_weights() const { return true; }
    inline int num_output() const { return num_output_; }
    inline int kernel_size_h() const { return kernel_size_h_; }
    inline int kernel_size_w() const { return kernel_size_w_; }
    inline int pad_h() const { return pad_h_; }
    inline int pad_w() const { return pad_w_; }
    inline int stride_h() const { return stride_h_; }
    inline int stride_w() const { return stride_w_; }
    inline int dilation_h() const { return dilation_h_; }
    inline int dilation_w() const { return dilation_w_; }
    inline int group() const { return group_; }
    inline bool has_bias_term() const { return has_bias_term_; }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"ConvlutionLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

    inline bool CheckWeights() const {
      if (this->weights_.empty() ||
	  (this->weights_.size() == 1 && this->has_bias_term()) ||
	  this->weights_.size() > 2) {
	BOOST_LOG_TRIVIAL(error)<<"ConvlutionLayer: Invalid weights or weights have not been imported.";
	return false;
      }
      return true;
    }

    void ComputeOutputShape(const std::vector<int>& input_shape,
			    std::vector<int>& output_shape);

  protected:
    int num_output_;
    int kernel_size_h_, kernel_size_w_;
    int pad_h_, pad_w_;
    int stride_h_, stride_w_;
    int dilation_h_, dilation_w_;

    bool has_bias_term_;
    
    int group_;

    DISABLE_COPY_AND_ASSIGN(BaseConvLayer);
  }; // class BaseConvLayer

} // namespace facethink


#endif
