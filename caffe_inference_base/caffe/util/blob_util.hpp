#ifndef __FACETHINK_CAFFE_NETBUILDER_BLOB_UTIL_HPP__
#define __FACETHINK_CAFFE_NETBUILDER_BLOB_UTIL_HPP__

#include "caffe/core/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace facethink {

  template <typename Dtype>
  class EXPORT_CLASS BlobUtil {
  public:
    static void ReshapeFromProto(const caffe::BlobShape& blob_shape,std::shared_ptr<Blob<Dtype> >& blob);

    static void ImportFromProto(const caffe::BlobProto& proto,std::shared_ptr<Blob<Dtype> >& blob);
    
	static void ExportToProto(Blob<Dtype>& blob, std::shared_ptr<caffe::BlobProto>& proto);


    DISABLE_COPY_AND_ASSIGN(BlobUtil);
  }; // class BlobUtil

} // namespace facethink

#endif 
