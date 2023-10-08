#include "caffe/layers/PriorBox/priorbox_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void PriorBoxLayer<Dtype>::ForwardComputation() {
    const int layer_width = this->inputs_[0]->shape(2);
    const int layer_height = this->inputs_[0]->shape(3);

    int img_width, img_height;
    if (this->img_h_ == 0 || this->img_w_ == 0) {
      img_width = this->inputs_[1]->shape(2);
      img_height = this->inputs_[1]->shape(3);
    } else {
      img_width = this->img_w_;
      img_height = this->img_h_;
    }

    Dtype step_x, step_y;
    if (this->step_w_ == 0 || this->step_h_ == 0) {
      step_x = static_cast<Dtype>(img_width) / layer_width;
      step_y = static_cast<Dtype>(img_height) / layer_height;
    } else {
      step_x = this->step_w_;
      step_y = this->step_h_;
    }


    const Dtype* ar_data = this->aspect_ratios_blob_.cpu_data();
    int ar_count = this->aspect_ratios_blob_.count();
    const Dtype* var_data = this->variances_blob_.cpu_data();
    int var_count = this->variances_blob_.count();
    
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    int dim = layer_height * layer_width * this->num_priors_ * 4;
    int idx = 0;
    for (int h = 0; h < layer_height; ++h) {
      for (int w = 0; w < layer_width; ++w) {
	Dtype center_x = (w + this->offset_) * step_x;
	Dtype center_y = (h + this->offset_) * step_y;
	Dtype box_width, box_height;
	// first prior: aspect_ratio = 1, size = min_size
	box_width = box_height = this->min_size_;	
	// xmin
	output_data[idx++] = (center_x - box_width / 2.) / img_width;
	// ymin
	output_data[idx++] = (center_y - box_height / 2.) / img_height;
	// xmax
	output_data[idx++] = (center_x + box_width / 2.) / img_width;
	// ymax
	output_data[idx++] = (center_y + box_height / 2.) / img_height;

	if (this->max_size_ > 0) {
	  // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
	  box_width = box_height = sqrt(this->min_size_ * this->max_size_);
	  // xmin
	  output_data[idx++] = (center_x - box_width / 2.) / img_width;
	  // ymin
	  output_data[idx++] = (center_y - box_height / 2.) / img_height;
	  // xmax
	  output_data[idx++] = (center_x + box_width / 2.) / img_width;
	  // ymax
	  output_data[idx++] = (center_y + box_height / 2.) / img_height;
	}

	// rest of priors
	for (int r = 0; r < ar_count; ++r) {
	  Dtype ar = ar_data[r];
	  if (fabs(ar - Dtype(1.0)) < 1e-6) {
	    continue;
	  }
	  box_width = this->min_size_ * sqrt(ar);
	  box_height = this->min_size_ / sqrt(ar);
	  // xmin
	  output_data[idx++] = (center_x - box_width / 2.) / img_width;
	  // ymin
	  output_data[idx++] = (center_y - box_height / 2.) / img_height;
	  // xmax
	  output_data[idx++] = (center_x + box_width / 2.) / img_width;
	  // ymax
	  output_data[idx++] = (center_y + box_height / 2.) / img_height;
	}
      }
    }
    
    // clip the prior's coordidate such that it is within [0, 1]
    if (this->clip_) {
      for (int d = 0; d < dim; ++d) {
	output_data[d] = std::min<Dtype>(std::max<Dtype>(output_data[d], Dtype(0.0)), Dtype(1.0));
      }
    }
    // set the variance.
    output_data += dim;
    if (var_count == 1) {
      caffe_cpu_set<Dtype>(dim, Dtype(var_data[0]), output_data);
    } else {
      int count = 0;
      for (int h = 0; h < layer_height; ++h) {
	for (int w = 0; w < layer_width; ++w) {
	  for (int i = 0; i < this->num_priors_; ++i) {
	    for (int j = 0; j < 4; ++j) {
	      output_data[count] = var_data[j];
	      ++count;
	    }
	  }
	}
      }
    }
  }

  INSTANTIATE_CLASS(PriorBoxLayer);
}
