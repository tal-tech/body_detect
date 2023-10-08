#include "caffe/layers/PriorBox/priorbox_layer_cuda.hpp"

namespace facethink {

  template <typename Dtype>
  __global__ void BoxLocKernel(const int nthreads, 
			       const int layer_height, const int layer_width,
			       const int img_height, const int img_width,
			       Dtype min_size,
			       Dtype max_size,
			       const Dtype* aspect_ratios,
			       const int aspect_ratios_size,
			       Dtype step_w, Dtype step_h,
			       Dtype offset,
			       Dtype* out){
    CUDA_KERNEL_LOOP(index, nthreads){
      int offset_iter = 0;
      if (max_size > 0){
	offset_iter = aspect_ratios_size + 1;
      }else{
	offset_iter = aspect_ratios_size;
      }
      
      int rindex = index * offset_iter * 4;

      int w = index % layer_width;
      int h = index / layer_width;

      Dtype center_x = (w + offset) * step_w;
      Dtype center_y = (h + offset) * step_h;
      
      Dtype box_width = min_size;
      Dtype box_height = min_size;
      out[rindex++] = (center_x - box_width  / Dtype(2.0)) / img_width;
      // ymin
      out[rindex++] = (center_y - box_height / Dtype(2.0)) / img_height;
      // xmax
      out[rindex++] = (center_x + box_width  / Dtype(2.0)) / img_width;
      // ymax
      out[rindex++] = (center_y + box_height / Dtype(2.0)) / img_height;

      if (max_size > 0){
	box_width = sqrt(min_size * max_size);
	box_height = box_width;

	out[rindex++] = (center_x - box_width  / Dtype(2.0)) / img_width;
	// ymin
	out[rindex++] = (center_y - box_height / Dtype(2.0)) / img_height;
	// xmax
	out[rindex++] = (center_x + box_width  / Dtype(2.0)) / img_width;
	// ymax
	out[rindex++] = (center_y + box_height / Dtype(2.0)) / img_height;	
      }

      // rest of priors
      for (int r = 0; r < aspect_ratios_size; ++r) {
	Dtype ar = aspect_ratios[r];
	if (fabs(ar - Dtype(1.0)) < 1e-6){
          continue;
        }
	box_width =  min_size * sqrt(ar);
	box_height = min_size / sqrt(ar);
	out[rindex++] = (center_x - box_width  / Dtype(2.0)) / img_width;
	// ymin
	out[rindex++] = (center_y - box_height / Dtype(2.0)) / img_height;
	// xmax
	out[rindex++] = (center_x + box_width  / Dtype(2.0)) / img_width;
	// ymax
	out[rindex++] = (center_y + box_height / Dtype(2.0)) / img_height;	
      }
    }  
  }

  
  template <typename Dtype>
  __global__ void ClipKernel(const int nthreads, Dtype* A){
    CUDA_KERNEL_LOOP(index, nthreads){
      if (A[index] < Dtype(0.)){
	A[index] = Dtype(0.);
      }

      if (A[index] > Dtype(1.0)){
	A[index] = Dtype(1.0);
      }
    }
  }

  template <typename Dtype>
  __global__ void VarKernel(const int nthreads,
			    const Dtype* variances, const int var_size, Dtype* A){
    CUDA_KERNEL_LOOP(index, nthreads){
      if (var_size == 1){
	for (int vv = 0; vv<4; ++vv){
	  A[index*4 + vv] = variances[0];
	}
      }else if (var_size == 4){
	for (int vv = 0; vv<4; ++vv){
	  A[index*4 + vv] = variances[vv];
	}
      }
    }
  }

  template <typename Dtype>
  void CUDAPriorBoxLayer<Dtype>::ForwardComputation(){
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

    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    
    int dim = layer_height * layer_width * this->num_priors_ * 4;
    
    const Dtype* ar_data = this->aspect_ratios_blob_.gpu_data();
    int ar_count = this->aspect_ratios_blob_.count();
    const Dtype* var_data = this->variances_blob_.gpu_data();
    int var_count = this->variances_blob_.count();

    int num_iter = layer_height * layer_width;
    BoxLocKernel<Dtype><<<CAFFE_GET_BLOCKS(num_iter), CAFFE_CUDA_NUM_THREADS>>>(
     										num_iter,
     										layer_height,
										layer_width,
     										img_height,
										img_width,
     										this->min_size_,
     										this->max_size_,
     										ar_data,
										ar_count,
     										step_x,
										step_y,
										this->offset_,
     										output_data
     										);
    CUDA_POST_KERNEL_CHECK;

    if (this->clip_){
      ClipKernel<Dtype><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(dim, output_data);
      CUDA_POST_KERNEL_CHECK;
    }

    output_data += dim;

    int num_iter2 =  layer_height * layer_width * this->num_priors_;
    VarKernel<Dtype><<<CAFFE_GET_BLOCKS(num_iter2), CAFFE_CUDA_NUM_THREADS>>>(num_iter2,
     									      var_data,
     									      var_count,
     									      output_data);
    CUDA_POST_KERNEL_CHECK;
    
  }
  
  INSTANTIATE_LAYER_GPU_FORWARD(CUDAPriorBoxLayer);
} // namespace facethink
