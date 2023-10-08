#include "caffe/util/bbox_util.hpp"

namespace facethink {

  template <typename Dtype>
  void PermuteDataCPU(const int count,
		      const Dtype* data, const int num_classes, const int num_data,
		      const int num_dim, Dtype* new_data) {
    for (int index = 0; index < count; ++index) {
      const int i = index % num_dim;
      const int c = (index / num_dim) % num_classes;
      const int d = (index / num_dim / num_classes) % num_data;
      const int n = index / num_dim / num_classes / num_data;
      const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
      new_data[new_index] = data[index];
    }
  }

  template void PermuteDataCPU(const int count,
			       const float* data, const int num_classes, const int num_data,
			       const int num_dim, float* new_data);

  ///////////////////////////////////////////////////////////////////////////////////////
  template <typename Dtype>
  void DecodeBBoxesCPU(const int count,
		       const Dtype* loc_data, const Dtype* prior_data,
		       const CodeType code_type, const bool variance_encoded_in_target,
		       const int num_priors, const bool share_location,
		       const int num_loc_classes, const int background_label_id,
		       const bool clip_bbox, Dtype* bbox_data) {
    for (int index = 0; index < count; ++index) {
      const int i = index % 4;
      const int c = (index / 4) % num_loc_classes;
      const int d = (index / 4 / num_loc_classes) % num_priors;
      if (!share_location && c == background_label_id) {
	// Ignore background class if not share_location.
	return;
      }
      const int pi = d * 4;
      const int vi = pi + num_priors * 4;
      if (code_type == CORNER) {
	if (variance_encoded_in_target) {
	  // variance is encoded in target, we simply need to add the offset
	  // predictions.
	  bbox_data[index] = prior_data[pi + i] + loc_data[index];
	} else {
	  // variance is encoded in bbox, we need to scale the offset accordingly.
	  bbox_data[index] =
	    prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
	}
      } else if (code_type == CENTER_SIZE) {
	const Dtype p_xmin = prior_data[pi];
	const Dtype p_ymin = prior_data[pi + 1];
	const Dtype p_xmax = prior_data[pi + 2];
	const Dtype p_ymax = prior_data[pi + 3];
	const Dtype prior_width = p_xmax - p_xmin;
	const Dtype prior_height = p_ymax - p_ymin;
	const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
	const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

	const Dtype xmin = loc_data[index - i];
	const Dtype ymin = loc_data[index - i + 1];
	const Dtype xmax = loc_data[index - i + 2];
	const Dtype ymax = loc_data[index - i + 3];

	Dtype decode_bbox_center_x, decode_bbox_center_y;
	Dtype decode_bbox_width, decode_bbox_height;
	if (variance_encoded_in_target) {
	  // variance is encoded in target, we simply need to retore the offset
	  // predictions.
	  decode_bbox_center_x = xmin * prior_width + prior_center_x;
	  decode_bbox_center_y = ymin * prior_height + prior_center_y;
	  decode_bbox_width = exp(xmax) * prior_width;
	  decode_bbox_height = exp(ymax) * prior_height;
	} else {
	  // variance is encoded in bbox, we need to scale the offset accordingly.
	  decode_bbox_center_x =
	    prior_data[vi] * xmin * prior_width + prior_center_x;
	  decode_bbox_center_y =
	    prior_data[vi + 1] * ymin * prior_height + prior_center_y;
	  decode_bbox_width =
	    exp(prior_data[vi + 2] * xmax) * prior_width;
	  decode_bbox_height =
	    exp(prior_data[vi + 3] * ymax) * prior_height;
	}

	switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
	}
      } else if (code_type == CORNER_SIZE) {
	const Dtype p_xmin = prior_data[pi];
	const Dtype p_ymin = prior_data[pi + 1];
	const Dtype p_xmax = prior_data[pi + 2];
	const Dtype p_ymax = prior_data[pi + 3];
	const Dtype prior_width = p_xmax - p_xmin;
	const Dtype prior_height = p_ymax - p_ymin;
	Dtype p_size;
	if (i == 0 || i == 2) {
	  p_size = prior_width;
	} else {
	  p_size = prior_height;
	}
	if (variance_encoded_in_target) {
	  // variance is encoded in target, we simply need to add the offset
	  // predictions.
	  bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
	} else {
	  // variance is encoded in bbox, we need to scale the offset accordingly.
	  bbox_data[index] =
	    prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
	}
      } else {
	// Unknown code type.
      }
      if (clip_bbox) {
	bbox_data[index] = std::max(std::min(bbox_data[index], Dtype(1.)), Dtype(0.));
      }
    }
  }

  template void DecodeBBoxesCPU(const int count,
				const float* loc_data, const float* prior_data,
				const CodeType code_type, const bool variance_encoded_in_target,
				const int num_priors, const bool share_location,
				const int num_loc_classes, const int background_label_id,
				const bool clip_bbox, float* bbox_data);

  ////////////////////////////////////////////////////////////////////////////////////////
  
  template <typename Dtype>
  void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
			const int top_k, std::vector<std::pair<Dtype, int> >* score_index_vec) {
    // Generate index score pairs.
    for (int i = 0; i < num; ++i) {
      if (scores[i] > threshold) {
	score_index_vec->push_back(std::make_pair(scores[i], i));
      }
    }

    // Sort the score pair according to the scores in descending order
    std::sort(score_index_vec->begin(), score_index_vec->end(),
	      SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_index_vec->size()) {
      score_index_vec->resize(top_k);
    }
  }

  template
  void GetMaxScoreIndex(const float* scores, const int num, const float threshold,
			const int top_k, std::vector<std::pair<float, int> >* score_index_vec);
 

  //////////////////////////////////////////////////////////////////////////////////////////////
  template <typename Dtype>
  Dtype BBoxSize(const Dtype* bbox, const bool normalized) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
      // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
      return Dtype(0.);
    } else {
      const Dtype width = bbox[2] - bbox[0];
      const Dtype height = bbox[3] - bbox[1];
      if (normalized) {
	return width * height;
      } else {
	// If bbox is not within range [0, 1].
	return (width + 1) * (height + 1);
      }
    }
  }

  template float BBoxSize(const float* bbox, const bool normalized);
  //////////////////////////////////////////////////////////////////////////////////////////////

  
  template <typename Dtype>
  Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
	bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
      return Dtype(0.);
    } else {
      const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
      const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
      const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
      const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

      const Dtype inter_width = inter_xmax - inter_xmin;
      const Dtype inter_height = inter_ymax - inter_ymin;
      const Dtype inter_size = inter_width * inter_height;

      const Dtype bbox1_size = BBoxSize(bbox1);
      const Dtype bbox2_size = BBoxSize(bbox2);

      return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
  }

  template float JaccardOverlap(const float* bbox1, const float* bbox2);

  //////////////////////////////////////////////////////////////////////////////////////////
  
  template <typename Dtype>
  void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
		    const float score_threshold, const float nms_threshold,
		    const float eta, const int top_k, std::vector<int>* indices) {
    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<Dtype, int> > score_index_vec;
    GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    while (score_index_vec.size() != 0) {
      const int idx = score_index_vec.front().second;
      bool keep = true;
      for (int k = 0; k < indices->size(); ++k) {
	if (keep) {
	  const int kept_idx = (*indices)[k];
	  float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
	  keep = overlap <= adaptive_threshold;
	} else {
	  break;
	}
      }
      if (keep) {
	indices->push_back(idx);
      }
      score_index_vec.erase(score_index_vec.begin());
      if (keep && eta < 1 && adaptive_threshold > 0.5) {
	adaptive_threshold *= eta;
      }
    }
  }

  template
  void ApplyNMSFast(const float* bboxes, const float* scores, const int num,
		    const float score_threshold, const float nms_threshold,
		    const float eta, const int top_k, std::vector<int>* indices);

  ////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  bool SortScorePairDescend(const std::pair<float, T>& pair1,
			    const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
  }

  // Explicit initialization.
  template bool SortScorePairDescend(const std::pair<float, int>& pair1,
				     const std::pair<float, int>& pair2);
  template bool SortScorePairDescend(const std::pair<float, std::pair<int, int> >& pair1,
				     const std::pair<float, std::pair<int, int> >& pair2);

} // namespace facethink
