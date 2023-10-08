#ifndef __FACETHINK_CAFFE_LAYERS_DETECTION_OUTPUT_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_DETECTION_OUTPUT_LAYER_CPU_HPP__

#include "caffe/layers/DetectionOutput/detection_output_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class DetectionOutputLayer: public BaseDetectionOutputLayer<Dtype> {
  public:
    explicit DetectionOutputLayer(int num_classes,
				  float nms_threshold, int top_k,
				  float confidence_threshold, int keep_top_k,
				  bool share_location = true,
				  int background_label_id = 0,
				  CodeType code_type = CodeType::CENTER_SIZE,
				  bool variance_encoded_in_target = false,
				  const std::string& name = "")
      :BaseDetectionOutputLayer<Dtype> (num_classes,
					nms_threshold,
					top_k,
					confidence_threshold,
					keep_top_k,
					share_location,
					background_label_id,
					code_type,
					variance_encoded_in_target,
					name) {
    }

    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(DetectionOutputLayer);

  }; // class DetectionOutputLayer

} // namespace facethink


#endif
