#include "caffe/layers/InnerProduct/inner_product_layer_base.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseInnerProductLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    
    int axis = this->inputs_[0]->CanonicalAxisIndex(axis_);

    N_ = num_output_;
    M_ = this->inputs_[0]->count(0, axis);
    K_ = this->inputs_[0]->count(axis);

    //std::cout<<N_<<"   "<<M_<<"   "<<K_<<std::endl;
   
    std::vector<int> output_shape = this->inputs_[0]->shape();
    output_shape.resize(axis + 1);
    output_shape[axis] = num_output_;
    
    this->outputs_[0]->Reshape(output_shape);


    if (has_bias_term()) {
      bias_multiplier_.Reshape( std::vector<int> { M_ } );
      caffe_cpu_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
  }

  INSTANTIATE_CLASS(BaseInnerProductLayer);
}
