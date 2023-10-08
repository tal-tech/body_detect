///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: det_body_ssd.hpp
///  @brief 检测图像中的人身框
///  @details 
///
///
///  @version 2.0.2.1
///  @author Ye Gensheng
///  @date 2018-05-07
///
///  @see 使用参考：demo.cpp
///
///////////////////////////////////////////////////////////////////////////////////////

#ifndef __FACETHINK_API_DET_BODY_SSD_HPP__
#define __FACETHINK_API_DET_BODY_SSD_HPP__

#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>

#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS   __declspec(dllexport)
#define EXPORT_API  extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API

#else
#define EXPORT_CLASS   __declspec(dllimport )
#define EXPORT_API  extern "C" __declspec(dllimport )
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API  extern "C" __attribute__((visibility("default")))   
#define EXPORT_CLASS_API __attribute__((visibility("default")))   
#endif

namespace facethink {

	class EXPORT_CLASS BodyDetectionSSD {
	public:

		EXPORT_CLASS_API explicit BodyDetectionSSD(void);
		EXPORT_CLASS_API virtual ~BodyDetectionSSD(void);

		/*!
		\brief SDK初始化函数，必须先于任何其他SDK函数之前调用。
		@param [in] det_model_prototxt 指定SDK对应的模型网络文件路径。
		@param [in] det_model_binary 指定SDK对应的模型参数文件路径。
		@param [in] config_file 指定SDK对应的参数配置文件路径。
		@param [in] gpu_id 指定GPU ID(仅GPU版本有效)。
		@return
		@remarks 初始化函数需要读取模型等文件，需要一定时间等待。
		*/
		EXPORT_CLASS_API static BodyDetectionSSD* create(
			const std::string& det_model_prototxt,
			const std::string& det_model_binary,
			const std::string& config_file, 
			const int gpu_id = 0);

		/*!
		\brief SDK初始化函数，必须先于任何其他SDK函数之前调用，create的重载函数。
		@param [in] det_model_file 指定SDK对应的模型文件路径。
		@param [in] config_file 指定SDK对应的参数配置文件路径,详情见config.ini文件。
		@param [in] gpu_id 指定GPU ID(仅GPU版本有效)。
		@return
		@remarks 初始化函数需要读取模型等文件，需要一定时间等待。
		*/
		EXPORT_CLASS_API static BodyDetectionSSD* create(
			const std::string& det_model_file,
			const std::string& config_file,
			const int gpu_id = 0);


		/// \brief 人身体框检测。
		/// @param [in] img 输入的图像数据，仅支持如下一种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] rectangles 保存SDK检测到的身体矩形框，容器中每个元素表示一个人的检测结果。
		/// @param [out] confidence 身体矩形框检测的置信度，范围（0-1），其中每个元素与rectangles中的元素对应。
		/// @return
		EXPORT_CLASS_API virtual void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence) = 0;

	};

}

#endif
