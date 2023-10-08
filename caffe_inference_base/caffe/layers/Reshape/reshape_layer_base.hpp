#ifndef __FACETHINK_CAFFE_LAYERS_RESHAPE_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_RESHAPE_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseReshapeLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseReshapeLayer(const std::vector<int>& dims,
			      int axis,
			      int num_axes,
			      const std::string& name="")
      :BaseLayer<Dtype>(name),
      dims_(dims), axis_(axis), num_axes_(num_axes) {

      InitLayer();
    }

    virtual void InitLayer();
    virtual void ForwardShape() ;
    virtual void ForwardComputation() {};

    virtual std::string type() const {
      return "Reshape";
    }

    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", shape: [";
      for (size_t i = 0; i < dims_.size(); ++i) {
	stream<<dims_.at(i)<<",";
      }
      stream<<"]";
      stream<<", axis: "<<axis_
	    <<", num_axis: "<<num_axes_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

  protected:
    std::vector<int> dims_;
    int axis_;
    int num_axes_;

    /// @brief vector of axes indices whose dimensions we'll copy from the bottom
    std::vector<int> copy_axes_;
    /// @brief the index of the axis whose dimension we infer, or -1 if none
    int inferred_axis_;
    /// @brief the product of the "constant" output dimensions
    int constant_count_;

    DISABLE_COPY_AND_ASSIGN(BaseReshapeLayer);
  }; // class BaseReshapeLayer

} // namespace facethink


#endif
