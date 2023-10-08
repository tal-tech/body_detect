#include "caffe/core/net.hpp"

namespace facethink {

  template <typename Dtype>
  void Net<Dtype>::Forward() {
    for (size_t i = 0; i<layers_.size(); ++i) {
      BOOST_LOG_TRIVIAL(debug)<<"Forwarding: "<<layers_.at(i)->type()<<
	"Layer ("<<layers_.at(i)->name()<<")";
      layers_[i]->Forward();
    }
  }


  template <typename Dtype>
  void Net<Dtype>::Forward(const RPNConfig<Dtype>& rpn_config) {
    for (size_t i = 0; i<layers_.size(); ++i) {
      BOOST_LOG_TRIVIAL(debug)<<"Forwarding: "<<layers_.at(i)->type()<<
	"Layer ("<<layers_.at(i)->name()<<")";

      if (!layers_[i]->type().compare("Proposal") ||
	  !layers_[i]->type().compare("Proposal(CUDA)")) {
	layers_[i]->ForwardWithConfig(rpn_config);
      } else {
	layers_[i]->Forward();
      }
    }
  }

  template <typename Dtype>
  void Net<Dtype>::AppendLayer(const std::shared_ptr<BaseLayer<Dtype> >& layer) {
    layers_.push_back(layer);
  }

  template <typename Dtype>
  void Net<Dtype>::InsertBlob(const std::string& blob_name, std::shared_ptr<Blob<Dtype> >& blob) {
    blobs_.insert( {blob_name, blob} );
  }

  INSTANTIATE_CLASS(Net);

} // namespace facethink
