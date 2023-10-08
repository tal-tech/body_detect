#ifndef __FACETHINK_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/util/bbox_util.hpp"

namespace facethink {

  template <typename Dtype>
  class DetectionPoseOutputLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& num_classes,
			      int& num_poses,
			      float& nms_threshold,
			      int& top_k,
			      float& confidence_threshold,
			      int& keep_top_k,
			      bool& share_location,
			      bool& share_pose,
			      int& background_label_id,
			      CodeType& code_type,
			      bool& variance_encoded_in_target,
			      std::string& name);

  }; // class DetectionPoseOutputLayerBuilder

  REGISTER_LAYER_BUILDER(DetectionPoseOutputLayerBuilder, DetectionPoseOutput);


} // namespace facethink


#endif
