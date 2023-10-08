#ifndef __FACETHINK_CAFFE_UTIL_BBOX_UTIL_HPP__
#define __FACETHINK_CAFFE_UTIL_BBOX_UTIL_HPP__

#include "caffe/core/common.hpp"

namespace facethink {
  
  enum CodeType { CORNER, CENTER_SIZE, CORNER_SIZE };

  
  template <typename Dtype>
  void PermuteDataCPU(const int count,
		      const Dtype* data, const int num_classes, const int num_data,
		      const int num_dim, Dtype* new_data);
  

  
  template <typename Dtype>
  void DecodeBBoxesCPU(const int count,
		       const Dtype* loc_data, const Dtype* prior_data,
		       const CodeType code_type, const bool variance_encoded_in_target,
		       const int num_priors, const bool share_location,
		       const int num_loc_classes, const int background_label_id,
		       const bool clip_bbox, Dtype* bbox_data);
  
  // Do non maximum suppression based on raw bboxes and scores data.
  // Inspired by Piotr Dollar's NMS implementation in EdgeBox.
  // https://goo.gl/jV3JYS
  //    bboxes: an array of bounding boxes.
  //    scores: an array of corresponding confidences.
  //    num: number of total boxes/confidences in the array.
  //    score_threshold: a threshold used to filter detection results.
  //    nms_threshold: a threshold used in non maximum suppression.
  //    eta: adaptation rate for nms threshold (see Piotr's paper).
  //    top_k: if not -1, keep at most top_k picked indices.
  //    indices: the kept indices of bboxes after nms.
  template <typename Dtype>
  void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
		    const float score_threshold, const float nms_threshold,
		    const float eta, const int top_k, std::vector<int>* indices);

  template <typename Dtype>
  void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
			const int top_k, std::vector<std::pair<Dtype, int> >* score_index_vec);

  template <typename Dtype>
  Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2);

  template <typename Dtype>
  Dtype BBoxSize(const Dtype* bbox, const bool normalized = true);
  
  // Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
  // in descend order based on the score (first) value.
  template <typename T>
  bool SortScorePairDescend(const std::pair<float, T>& pair1,
			    const std::pair<float, T>& pair2);

#ifndef CPU_ONLY

  template <typename Dtype>
  void PermuteDataGPU(const int nthreads,
		      const Dtype* data, const int num_classes, const int num_data,
		      const int num_dim, Dtype* new_data);


  template <typename Dtype>
  void DecodeBBoxesGPU(const int nthreads,
		       const Dtype* loc_data, const Dtype* prior_data,
		       const CodeType code_type, const bool variance_encoded_in_target,
		       const int num_priors, const bool share_location,
		       const int num_loc_classes, const int background_label_id,
		       const bool clip_bbox, Dtype* bbox_data);

#endif 
} // namespace facethink

#endif
