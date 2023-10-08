#include "caffe/core/blob.hpp"
#include <cstring>

namespace facethink {

  template <typename Dtype>
  Blob<Dtype>::Blob(const std::vector<int>& shape)
    :data_(), count_(0), capacity_(0) {
    Reshape(shape);
  }

  template <typename Dtype>
  void Blob<Dtype>::Reshape(const std::vector<int>& shape) {
    count_ = 1;
    shape_.resize(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0) {
	BOOST_LOG_TRIVIAL(error)<<"Blob: Invalid Shape Size";
      }

      if (shape[i] >= std::numeric_limits<int>::max() / count_) {
	BOOST_LOG_TRIVIAL(error)<<"Blob: blob size exceeds INT_MAX";
      }

      count_ *= shape[i];
      shape_[i] = shape[i];
    }

    if (count_ > capacity_) {
      capacity_ = count_;
      data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    }
  }

  template <typename Dtype>
  void Blob<Dtype>::ReshapeLike(const Blob& other) {
    Reshape(other.shape());
  }


  template <typename Dtype>
  void Blob<Dtype>::ShareData(const Blob& other) {
    if (count_ != other.count()){
      BOOST_LOG_TRIVIAL(error)<<"Blob : can not share data, count not equal";
    }
    data_ = other.data();
  }

  template <typename Dtype>
  const Dtype* Blob<Dtype>::cpu_data() const {
    if (CheckData()) {
      return (const Dtype*)data_->cpu_data();
    }
    return nullptr;
  }

  template <typename Dtype>
  Dtype* Blob<Dtype>::mutable_cpu_data() {
    if (CheckData()) {
      return static_cast<Dtype*>(data_->mutable_cpu_data());
    }
    return nullptr;
  }

#ifndef CPU_ONLY
  template <typename Dtype>
  const Dtype* Blob<Dtype>::gpu_data() const {
    if (CheckData()) {
      return (const Dtype*)data_->gpu_data();
    }
    return nullptr;
  }

  template <typename Dtype>
  Dtype* Blob<Dtype>::mutable_gpu_data() {
    if (CheckData()) {
      return static_cast<Dtype*>(data_->mutable_gpu_data());
    }
    return nullptr;
  }
#endif


  // need detemine Blob shape first.
  template <typename Dtype>
  void Blob<Dtype>::ImportFromExtenalData(const Dtype* data_ext, int num) {
    if (count_ != num) {
      BOOST_LOG_TRIVIAL(error)<<"Blob: import extenal data, count mismatch ( not equal)"; 
    }

    if (!data_ext){
      BOOST_LOG_TRIVIAL(error)<<"Blob: import extenal data,  Invalid data nullptr";
    }else{
      Dtype* data_p = mutable_cpu_data();
      memcpy(data_p, data_ext, sizeof(Dtype)*num);
    }
  }

  template <typename Dtype>
  void Blob<Dtype>::ImportFrom(const std::vector<Dtype>& v) {
    this->Reshape( std::vector<int>{ static_cast<int>(v.size()) } );
    Dtype* bdata = this->mutable_cpu_data();
    for (size_t i = 0; i < v.size(); ++i) {
      bdata[i] = v[i];
    }
  }

  INSTANTIATE_CLASS(Blob);
  template class Blob<int>;
  
} // namespace facethink
