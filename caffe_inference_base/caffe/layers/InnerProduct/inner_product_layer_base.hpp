#ifndef __FACETHINK_CAFFE_LAYERS_INNER_PRODUCT_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_INNER_PRODUCT_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {
  
  template <typename Dtype>
  class BaseInnerProductLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseInnerProductLayer(int num_output,
				   bool has_bias_term = true,
				   int axis = 1,
				   bool transpose = false,
				   const std::string& name = "")
      :BaseLayer<Dtype>(name),
      num_output_(num_output),
      has_bias_term_(has_bias_term),
      axis_(axis),
      transpose_(transpose) {

    }

    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    // output convolution weights and bias
    virtual inline const std::shared_ptr<Blob<Dtype> > weight() {
      if (this->weights_.empty()){
	BOOST_LOG_TRIVIAL(error)<< "InnerProductLayer: No weight found.";
	return nullptr;
      }
      return this->weights_[0];
    }
    virtual inline const std::shared_ptr<Blob<Dtype> >  bias() {
      if (this->weights_.size()<2 || !has_bias_term_) {
	BOOST_LOG_TRIVIAL(error)<< "InnerProductLayer: No bias found.";
	return nullptr;
      }
      return this->weights_[1];
    }

    virtual inline std::string type() const {
      return "InnerProduct";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", num_output: "<<num_output_
	     <<", bias_term: "<<(has_bias_term_? "True":"False")
	     <<", axis: "<<axis_
	     <<", transpose: "<<(transpose_? "True":"False");
      return stream.str();
    }

    virtual inline bool has_weights() { return true; }
    inline int num_output() const { return num_output_; }
    inline bool has_bias_term() const { return has_bias_term_; }
    inline int axis() const { return axis_; }
    inline bool transpose() const { return transpose_; }
    
  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"InnerProductLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

    inline bool CheckWeights() const {
      if (this->weights_.empty() ||
	  (this->weights_.size() == 1 && this->has_bias_term()) ||
	  this->weights_.size() > 2) {
	BOOST_LOG_TRIVIAL(error)<<"InnerProductLayer: Invalid weights or weights have not been imported.";
	return false;
      }
      return true;
    }
    
    
  protected:
    int num_output_;
    bool has_bias_term_;
    int axis_;
    bool transpose_;

    int M_;
    int K_;
    int N_;

    Blob<Dtype> bias_multiplier_;

    DISABLE_COPY_AND_ASSIGN(BaseInnerProductLayer);
  }; // BaseInnerProductLayer

} // namespace facethink

#endif
