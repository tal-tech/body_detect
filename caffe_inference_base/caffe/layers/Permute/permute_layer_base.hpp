#ifndef __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {
  
  template <typename Dtype>
  class BasePermuteLayer: public BaseLayer<Dtype> {
  public:
    explicit BasePermuteLayer(const std::vector<int>& permute_order,
			      const std::string& name="")
      :BaseLayer<Dtype>(name),
      permute_order_(permute_order),
      need_permute_(true) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "Permute";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", orders: [";
      for (size_t i=0; i<permute_order_.size(); ++i){
	stream<<permute_order_.at(i)<<",";
      }
      stream<<"]";
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"PermuteLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    std::vector<int> permute_order_;
    
    bool need_permute_;
    Blob<int> permute_order_blob_;
    Blob<int> old_steps_;
    Blob<int> new_steps_;

    DISABLE_COPY_AND_ASSIGN(BasePermuteLayer);
  }; // class BasePermuteLayer

} // namespace facethink


#endif
