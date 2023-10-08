#ifndef __FACETHINK_CAFFE_LAYERS_LRN_LAYERS_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_LRN_LAYERS_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseLRNLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseLRNLayer(int size,
			  Dtype alpha,
			  Dtype beta,
			  Dtype k,
			  const std::string& name="")
      :BaseLayer<Dtype>(name),
      size_(size),
      alpha_(alpha),
      beta_(beta),
      k_(k) {
    }
    
    virtual void ForwardShape() = 0;
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "LRN";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<" size: "<<size_
	    <<", alpha: "<<alpha_
	    <<", beta: "<<beta_
	    <<", k: "<<k_;
      return stream.str();
    }


  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"LRNLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

  protected:
    int size_;
    Dtype alpha_, beta_, k_;

    DISABLE_COPY_AND_ASSIGN(BaseLRNLayer);
  }; // class BaseLRNLayer

} // namespace facethink

#endif
