#ifndef __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {
  
  template <typename Dtype>
  class BaseConcatLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseConcatLayer(int concat_axis,
			     const std::string& name)
      :BaseLayer<Dtype>(name),
      concat_axis_(concat_axis) {

    }

    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual inline std::string type() const {
      return "Concat";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", concat_axis: "<<concat_axis_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.empty() || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"ConcatLayer: only accept multiple (>=1) input blob and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    int concat_axis_, concat_axis_canonical_;
    int num_concats_;
    int concat_input_size_;

    DISABLE_COPY_AND_ASSIGN(BaseConcatLayer);
  }; // BaseConcatLayer

} //namespace facethink


#endif
