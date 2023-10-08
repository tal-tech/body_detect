#ifndef __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {
  
  enum EltwiseOp { PROD, SUM, MAX };
  
  template <typename Dtype>
  class BaseEltwiseLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseEltwiseLayer(EltwiseOp op,
			      const std::vector<Dtype> coeffs,
			      const std::string& name="")
      :BaseLayer<Dtype>(name),
      op_(op),
      coeffs_(coeffs) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "Eltwise";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", EltwiseOp: ";
      if (op_ == EltwiseOp::PROD) {
	stream << "Prod ";
      } else if (op_ == EltwiseOp::SUM) {
	stream << "Sum ";
      } else {
	stream << "Max ";
      }

      stream <<", coeff: [";
      for (size_t i = 0; i < coeffs_.size(); ++i) {
	stream<<coeffs_[i]<<",";
      }
      stream<<"]";
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.empty() || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"BaseEltwiseLayer: only accept multiple (>=1) input blob and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    EltwiseOp op_;
    std::vector<Dtype> coeffs_;

    Blob<int> max_idx_;
    
    DISABLE_COPY_AND_ASSIGN(BaseEltwiseLayer);
  }; // class BaseEltwiseLayer

} // namespace facethink


#endif
