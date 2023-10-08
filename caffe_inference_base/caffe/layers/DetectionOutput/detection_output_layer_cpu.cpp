#include "caffe/layers/DetectionOutput/detection_output_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"
#include <map>

namespace facethink {
  
  template <typename Dtype>
  void DetectionOutputLayer<Dtype>::ForwardComputation() {
    const int    loc_count  = this->inputs_[0]->count();
    const Dtype* loc_data   = this->inputs_[0]->cpu_data();
    const int    conf_count = this->inputs_[1]->count();
    const Dtype* conf_data  = this->inputs_[1]->cpu_data();
    const Dtype* prior_data = this->inputs_[2]->cpu_data();

    
    // Decode predictions.
    Dtype* bbox_data = this->bbox_preds_.mutable_cpu_data();
 
    const bool clip_bbox = true;
    DecodeBBoxesCPU<Dtype>(loc_count,
			   loc_data,
			   prior_data,
			   this->code_type_,
    			   this->variance_encoded_in_target_,
			   this->num_priors_,
			   this->share_location_,
     			   this->num_loc_classes_,
			   this->background_label_id_,
			   clip_bbox,
			   bbox_data);
    
    // Retrieve all decoded location predictions.
    const Dtype* bbox_cpu_data;
    if (!this->share_location_) {
      Dtype* bbox_permute_data = this->bbox_permute_.mutable_cpu_data();
      PermuteDataCPU<Dtype>(loc_count,
			    bbox_data,
			    this->num_loc_classes_,
			    this->num_priors_,
			    4,
			    bbox_permute_data);
      bbox_cpu_data = this->bbox_permute_.cpu_data();
    } else {
      bbox_cpu_data = this->bbox_preds_.cpu_data();
    }

    // Retrieve all confidences.
    Dtype* conf_permute_data = this->conf_permute_.mutable_cpu_data();
    PermuteDataCPU<Dtype>(conf_count,
			  conf_data,
			  this->num_classes_,
			  this->num_priors_,
			  1,
			  conf_permute_data);
    const Dtype* conf_cpu_data = this->conf_permute_.cpu_data();

    const int num = this->inputs_[0]->shape(0);
    int num_kept = 0;
    std::vector<std::map<int, std::vector<int> > > all_indices;
    for (int i = 0; i < num; ++i) {
      std::map<int, std::vector<int> > indices;
      int num_det = 0;
      const int conf_idx = i * this->num_classes_ * this->num_priors_;
      int bbox_idx;
      if (this->share_location_) {
	bbox_idx = i * this->num_priors_ * 4;
      } else {
	bbox_idx = conf_idx * 4;
      }
      for (int c = 0; c < this->num_classes_; ++c) {
	if (c == this->background_label_id_) {
	  // Ignore background class.
	  continue;
	}
	const Dtype* cur_conf_data = conf_cpu_data + conf_idx + c * this->num_priors_;
	const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
	if (!this->share_location_) {
	  cur_bbox_data += c * this->num_priors_ * 4;
	}
	ApplyNMSFast(cur_bbox_data,
		     cur_conf_data,
		     this->num_priors_,
		     this->confidence_threshold_,
		     this->nms_threshold_,
		     1.0,  //eta
		     this->top_k_,
		     &(indices[c]));
	num_det += indices[c].size();
      }
      
      if (this->keep_top_k_ > -1 && num_det > this->keep_top_k_) {
	std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
	for (std::map<int, std::vector<int> >::iterator it = indices.begin();
	     it != indices.end(); ++it) {
	  int label = it->first;
	  const std::vector<int>& label_indices = it->second;
	  for (int j = 0; j < label_indices.size(); ++j) {
	    int idx = label_indices[j];
	    float score = conf_cpu_data[conf_idx + label * this->num_priors_ + idx];
	    score_index_pairs.push_back(std::make_pair(score, std::make_pair(label, idx)));
	  }
	}
	// Keep top k results per image.
	std::sort(score_index_pairs.begin(), score_index_pairs.end(),
		  SortScorePairDescend<std::pair<int, int> >);
	score_index_pairs.resize(this->keep_top_k_);
	// Store the new indices.
	std::map<int, std::vector<int> > new_indices;
	for (int j = 0; j < score_index_pairs.size(); ++j) {
	  int label = score_index_pairs[j].second.first;
	  int idx = score_index_pairs[j].second.second;
	  new_indices[label].push_back(idx);
	}
	all_indices.push_back(new_indices);
	num_kept += this->keep_top_k_;
      } else {
	all_indices.push_back(indices);
	num_kept += num_det;
      }
      
    }
    
  
    Dtype *output_data, *prob_data;
    if (num_kept == 0) {
      BOOST_LOG_TRIVIAL(info)<<"DetectionOutLayer: Couldn't find any detections";
      this->outputs_[0]->Reshape(std::vector<int> { 1, 7 });
      caffe_cpu_set(this->outputs_[0]->count(), Dtype(-1), this->outputs_[0]->mutable_cpu_data());

      if (this->outputs_.size() > 1){
	this->outputs_[1]->Reshape(std::vector<int> { 1,  this->num_classes_ } );
	caffe_cpu_set(this->outputs_[1]->count(), Dtype(-1), this->outputs_[1]->mutable_cpu_data());
      }
      return;
    } else {
      std::vector<int> output_shape = { num_kept, 7 };
      this->outputs_[0]->Reshape(output_shape);
      output_data = this->outputs_[0]->mutable_cpu_data();

      // reshape probs
      if (this->outputs_.size() > 1){
	std::vector<int> prob_shape = { num_kept, this->num_classes_ };
	this->outputs_[1]->Reshape(prob_shape);
	prob_data = this->outputs_[1]->mutable_cpu_data();
      }
    }
 
    int count = 0;

    for (int i = 0; i < num; ++i) {
      const int conf_idx = i * this->num_classes_ * this->num_priors_;
      int bbox_idx;
      if (this->share_location_) {
	bbox_idx = i * this->num_priors_ * 4;
      } else {
	bbox_idx = conf_idx * 4;
      }
      
      for (std::map<int, std::vector<int> >::iterator it = all_indices[i].begin();
	   it != all_indices[i].end(); ++it) {
	int label = it->first;
	std::vector<int>& indices = it->second;
	const Dtype* cur_conf_data =
	  conf_cpu_data + conf_idx + label * this->num_priors_;
	const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
	if (!this->share_location_) {
	  cur_bbox_data += label * this->num_priors_ * 4;
	}
	
	for (int j = 0; j < indices.size(); ++j) {
	  int idx = indices[j];
	  output_data[count * 7] = i;
	  output_data[count * 7 + 1] = label;
	  output_data[count * 7 + 2] = cur_conf_data[idx];
	  for (int k = 0; k < 4; ++k) {
	    output_data[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
	  }
	  
	  // fill prob blob
	  if (this->outputs_.size() > 1){
	    for (int c = 0; c < this->num_classes_; ++c) {
	      prob_data[count*this->num_classes_ + c] = 
		conf_cpu_data[conf_idx + c * this->num_priors_ + idx];
	    }
	  }
	  ++count;
	}
      }
    }
  }

  INSTANTIATE_CLASS(DetectionOutputLayer);
}
