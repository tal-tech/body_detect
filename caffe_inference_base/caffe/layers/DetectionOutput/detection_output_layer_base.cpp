#include "caffe/layers/DetectionOutput/detection_output_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseDetectionOutputLayer<Dtype>::ForwardShape() {
    // input :
    // 0. loc 1. conf, 3. priorbox
    this->CheckBlobs();

    if (this->inputs_[0]->shape(0) != this->inputs_[1]->shape(0)){
      BOOST_LOG_TRIVIAL(error)<<"DetectionOutLayer: the number of location and confidence must be equal";
    }
    
    num_priors_ = this->inputs_[2]->shape(2) / 4;
    num_loc_classes_ = share_location_? 1 : num_classes_;

    if (num_priors_ * num_loc_classes_ * 4 != this->inputs_[0]->shape(1)){
      BOOST_LOG_TRIVIAL(error)<<"DetectionPoseOutLayer: Number of priors must match number of location predictions.";
    }
    if (num_priors_ * num_classes_ != this->inputs_[1]->shape(1)){
      BOOST_LOG_TRIVIAL(error)<<"DetectionPoseOutLayer: Number of priors must match number of confidence predictions.";
    }    

    bbox_preds_.Reshape(this->inputs_[0]->shape());
    if (!share_location_){
      bbox_permute_.Reshape(this->inputs_[0]->shape());
    }
    conf_permute_.Reshape(this->inputs_[1]->shape());
    
    // Each row is a 7 dimension vector, which stores
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    this->outputs_[0]->Reshape( std::vector<int> { 1, 7 } );

    if (this->outputs_.size() > 1){
      this->outputs_[1]->Reshape( std::vector<int> { 1, num_classes_ } );
    }else{
      BOOST_LOG_TRIVIAL(info)<<"DetectionOutLayer: no prob output.";
    }
  }
  
  INSTANTIATE_CLASS(BaseDetectionOutputLayer);
}
