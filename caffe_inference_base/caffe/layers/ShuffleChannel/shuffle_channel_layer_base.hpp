#ifndef __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseShuffleChannelLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseShuffleChannelLayer(int group = 1,
				     const std::string& name="")
      :BaseLayer<Dtype>(name),
      group_(group) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "ShuffleChannel";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", group: "<<group_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"BaseShuffleChannelLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }    

  protected:
    int group_;

    int num_,  feature_map_size_, sp_sz_;
    int group_row_, group_column_;
   
    
    DISABLE_COPY_AND_ASSIGN(BaseShuffleChannelLayer);
  }; // class BaseShuffleChannelLayer

} // namespace facethink


#endif
