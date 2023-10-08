
# Set body box model
CAFFE_MODEL_BOX=model/det_body_ssd_v1.0.3.bin

# Set test data
IMAGES_PATH=images/testing/

# Set test data
CONFIG_PATH=model/config.ini


libs/linux/gpu/performance_testing_GPU $CAFFE_MODEL_BOX $IMAGES_PATH $CONFIG_PATH




