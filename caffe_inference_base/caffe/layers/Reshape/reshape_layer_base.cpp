#include "caffe/layers/Reshape/reshape_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseReshapeLayer<Dtype>::InitLayer() {
    inferred_axis_ = -1;
    copy_axes_.clear();
    constant_count_ = 1;

    for (size_t i = 0; i < dims_.size(); ++i) {
      if (dims_[i] == 0){
	copy_axes_.push_back(i);
      }else if (dims_[i] == -1){
	if (inferred_axis_ == -1){
	  inferred_axis_ = i;	    
	}else{
	  BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: -1 dims; at most a single (1) value of -1 may be specified.";
	}
      }else{
	constant_count_ *= dims_[i];
      }
    } 
  }
  
  template <typename Dtype>
  void BaseReshapeLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const int input_start_axis = axis_;
    const int start_axis = (input_start_axis >= 0) ? input_start_axis :
      this->inputs_[0]->num_axes() + input_start_axis + 1;   // +1 ?

    if (start_axis < 0 || start_axis > this->inputs_[0]->num_axes()){
      BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: axis out of range";
    }
    
    const int num_axes = num_axes_;
    if (num_axes < -1){
      BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: num_axes must be >= 0, or -1 for all.";
    }
    
    const int end_axis = (num_axes == -1) ? this->inputs_[0]->num_axes() : (start_axis + num_axes);
    if (end_axis > this->inputs_[0]->num_axes()){
      BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: end_axis = axis + num_axes is out of range";
    }
    
    const int num_axes_replaced = end_axis - start_axis;
    const int num_axes_retained = this->inputs_[0]->num_axes() - num_axes_replaced;
    const int num_new_axes = dims_.size();
    
    std::vector<int> output_shape(num_axes_retained + num_new_axes);
    
    int output_shape_index = 0;
    for (int i = 0; i < start_axis; ++i) {
      output_shape[output_shape_index++] = this->inputs_[0]->shape(i);
    }
    for (int i = 0; i < num_new_axes; ++i) {
      output_shape[output_shape_index++] = dims_[i];
    }
    for (int i = end_axis; i < this->inputs_[0]->num_axes(); ++i) {
      output_shape[output_shape_index++] = this->inputs_[0]->shape(i);
    }

    if (output_shape_index != output_shape.size()){
      BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: algorithm error.";
    }
    
    for (int i = 0; i < copy_axes_.size(); ++i) {
      const int copy_axis_index = copy_axes_[i];
      if (this->inputs_[0]->num_axes() <= start_axis + copy_axis_index){
	BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: new shape contains a 0, but there was no corresponding bottom axis to copy.";
      }
      output_shape[start_axis + copy_axis_index] =
        this->inputs_[0]->shape(start_axis + copy_axis_index);
    }

    if (inferred_axis_ >= 0) {
      // A -1 dim was specified; infer the correct dimension by computing the
      // product of the other dimensions.
      int explicit_count = constant_count_;
      explicit_count *= this->inputs_[0]->count(0, start_axis);
      explicit_count *= this->inputs_[0]->count(end_axis);
      for (int i = 0; i < copy_axes_.size(); ++i) {
	const int copy_axis_index = copy_axes_[i];
	explicit_count *= output_shape[start_axis + copy_axis_index];
      }
      if (this->inputs_[0]->count() % explicit_count != 0){
	BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: bottom count must be divisible by the product of the specified dimensions.";
      }
      const int inferred_dim = this->inputs_[0]->count() / explicit_count;
      output_shape[start_axis + inferred_axis_] = inferred_dim;
    }


    this->outputs_[0]->Reshape(output_shape);
    
    if (this->outputs_[0]->count() != this->inputs_[0]->count()){
      BOOST_LOG_TRIVIAL(error)<<"ReshapeLayer: output count must match input count.";
    }
    this->outputs_[0]->ShareData(*this->inputs_[0]);
  }

  INSTANTIATE_CLASS(BaseReshapeLayer);
  
} // namespace facethink 
