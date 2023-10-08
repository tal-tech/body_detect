#include "caffe/layers/Permute/permute_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BasePermuteLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    int num_axes =  this->inputs_[0]->num_axes();
    std::vector<int> orders;

    for (int i = 0; i < permute_order_.size(); ++i) {
      int order = permute_order_[i];
      if (order >= num_axes){
        BOOST_LOG_TRIVIAL(error)<< "PermuteLayer: order should be less than the input dimension.";
      }
      if (std::find(orders.begin(), orders.end(), order) != orders.end()){
	BOOST_LOG_TRIVIAL(error) << "PermuteLayer: duplicate orders found.";
      }
      orders.push_back(order);
    }
    // Push the rest orders. And save original step sizes for each axis.
    for (int i = 0; i < num_axes; ++i) {
      if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
	orders.push_back(i);
      }
    }

    if (num_axes != orders.size()) {
      BOOST_LOG_TRIVIAL(error)<<"PermuteLayer: permute size mismatch"; 
    }
    
    need_permute_ = false;
    for (size_t i = 0; i < orders.size(); ++i){
      if (orders[i] != i){
	need_permute_ = true;
	break;
      }
    }
    
    if (!need_permute_){
      this->outputs_[0]->Reshape(input_shape);
      this->outputs_[0]->ShareData(*this->inputs_[0]);
    }else{
      std::vector<int> output_shape(input_shape.size());
      for (size_t i=0; i<input_shape.size(); ++i){
	output_shape[i] = input_shape[orders[i]];
      }
      this->outputs_[0]->Reshape(output_shape);

      int count = this->inputs_[0]->count();

      permute_order_blob_.Reshape(std::vector<int>{ num_axes });
      old_steps_.Reshape(std::vector<int> { num_axes });
      new_steps_.Reshape(std::vector<int> { num_axes });

      for (int i = 0; i < num_axes; ++i) {
	permute_order_blob_.mutable_cpu_data()[i] = orders[i];
      }
      
      for (int i = 0; i < num_axes; ++i) {
	if (i == 0){
	  old_steps_.mutable_cpu_data()[i] = count / input_shape[i];
	}else{
	  old_steps_.mutable_cpu_data()[i] = old_steps_.cpu_data()[i-1] / input_shape[i];
	}
      } 

      for (size_t i = 0; i < num_axes; ++i) {
	if (i == 0){
	  new_steps_.mutable_cpu_data()[i] = count / output_shape[i];
	}else{
	  new_steps_.mutable_cpu_data()[i] = new_steps_.cpu_data()[i-1] / output_shape[i];
	}
      }
    }
  }

  INSTANTIATE_CLASS(BasePermuteLayer);
} // namespace facethink
