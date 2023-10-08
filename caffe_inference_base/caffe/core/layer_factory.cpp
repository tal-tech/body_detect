#include "caffe/core/layer_factory.hpp"
#include "caffe/core/base_layer_builder.hpp"

namespace facethink {

  template LayerFactory<float>& LayerFactory<float>::instance();
  
} // namespace facethink
