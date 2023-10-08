#ifndef __FACETHINK_CAFFE_LAYERS_SOFTMAX_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_SOFTMAX_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseSoftmaxLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseSoftmaxLayer(int softmax_axis,
			      const std::string& name = "")
      :BaseLayer<Dtype>(name),
      softmax_axis_(softmax_axis) {
    }

    virtual void ForwardShape() = 0;
    virtual void ForwardComputation() = 0;
    
    virtual inline std::string type() const {
      return "Softmax";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", axis: "<<softmax_axis_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"SoftmaxLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    int softmax_axis_;

    DISABLE_COPY_AND_ASSIGN(BaseSoftmaxLayer);
  }; // class BaseSoftaxLayer
  
} // namespace facethink



#endif
