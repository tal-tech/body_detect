#include "caffe/util/blob_util.hpp"

namespace facethink {

  template <typename Dtype>
  void BlobUtil<Dtype>::ReshapeFromProto(const caffe::BlobShape& blob_shape,
					 std::shared_ptr<Blob<Dtype> >& blob) {
    std::vector<int> shape_vec(blob_shape.dim_size());
    for (int i = 0; i < blob_shape.dim_size(); ++i) {
      shape_vec[i] = blob_shape.dim(i);
    }
    blob->Reshape(shape_vec);
  }

  template <typename Dtype>
  void BlobUtil<Dtype>::ImportFromProto(const caffe::BlobProto& proto,
					std::shared_ptr<Blob<Dtype> >& blob) {
    if (blob == nullptr) {
      BOOST_LOG_TRIVIAL(error)<<"BlobUtil: Blob is a nullptr.";
    }
    
    std::vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
	proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    }else{
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i){
	shape[i] = proto.shape().dim(i);
      }
    }
    blob->Reshape(shape);

    // copy data
    if (blob->count() != proto.data_size()){
      BOOST_LOG_TRIVIAL(error)<<"BlobUtil: Blob count mismatch ( not equal)";
    }

    Dtype* data_vec = blob->mutable_cpu_data();
    int count = blob->count();
    for (int i = 0; i < count; ++i) {
      data_vec[i] = proto.data(i);
    }
  }

  template <typename Dtype>
  void BlobUtil<Dtype>::ExportToProto(Blob<Dtype>& blob, std::shared_ptr<caffe::BlobProto>& proto) {

	  proto->clear_shape();
	  std::vector<int> shape = blob.shape();
	  for (int i = 0; i < shape.size(); ++i) {
		  proto->mutable_shape()->add_dim(shape[i]);
	  }
	  proto->clear_data();
	  const float* data_vec = blob.cpu_data();
	  int count = blob.count();
	  for (int i = 0; i < count; ++i) {
		  proto->add_data(data_vec[i]);
	  }


  }

  INSTANTIATE_CLASS(BlobUtil);
  
} // namespace facethink
