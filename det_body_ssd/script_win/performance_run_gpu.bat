@echo off

::det body model
SET CAFFE_MODEL=model/det_body_ssd_v1.0.3.bin

::Set test data
SET IMAGES_PATH=images/testing/

SET CONFIG_PATH=model/config.ini

start libs/x64/gpu/performance_testing_GPU.exe ^
%CAFFE_MODEL% ^
%IMAGES_PATH% ^
%CONFIG_PATH%
