///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: det_body_ssd.hpp
///  @brief ���ͼ���е������
///  @details 
///
///
///  @version 2.0.2.1
///  @author Ye Gensheng
///  @date 2018-05-07
///
///  @see ʹ�òο���demo.cpp
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
		\brief SDK��ʼ�����������������κ�����SDK����֮ǰ���á�
		@param [in] det_model_prototxt ָ��SDK��Ӧ��ģ�������ļ�·����
		@param [in] det_model_binary ָ��SDK��Ӧ��ģ�Ͳ����ļ�·����
		@param [in] config_file ָ��SDK��Ӧ�Ĳ��������ļ�·����
		@param [in] gpu_id ָ��GPU ID(��GPU�汾��Ч)��
		@return
		@remarks ��ʼ��������Ҫ��ȡģ�͵��ļ�����Ҫһ��ʱ��ȴ���
		*/
		EXPORT_CLASS_API static BodyDetectionSSD* create(
			const std::string& det_model_prototxt,
			const std::string& det_model_binary,
			const std::string& config_file, 
			const int gpu_id = 0);

		/*!
		\brief SDK��ʼ�����������������κ�����SDK����֮ǰ���ã�create�����غ�����
		@param [in] det_model_file ָ��SDK��Ӧ��ģ���ļ�·����
		@param [in] config_file ָ��SDK��Ӧ�Ĳ��������ļ�·��,�����config.ini�ļ���
		@param [in] gpu_id ָ��GPU ID(��GPU�汾��Ч)��
		@return
		@remarks ��ʼ��������Ҫ��ȡģ�͵��ļ�����Ҫһ��ʱ��ȴ���
		*/
		EXPORT_CLASS_API static BodyDetectionSSD* create(
			const std::string& det_model_file,
			const std::string& config_file,
			const int gpu_id = 0);


		/// \brief ��������⡣
		/// @param [in] img �����ͼ�����ݣ���֧������һ�ָ�ʽ:
		/// - 1.BGRͼ��imgΪһά���飬ÿ��Ԫ�أ��ֽڣ���ʾһ�����ص�ĵ�ͨ��ȡֵ����������Ԫ�ر�ʾһ�����ص����ͨ��ȡֵ��˳��ΪBGR��
		/// @param [out] rectangles ����SDK��⵽��������ο�������ÿ��Ԫ�ر�ʾһ���˵ļ������
		/// @param [out] confidence ������ο�������Ŷȣ���Χ��0-1��������ÿ��Ԫ����rectangles�е�Ԫ�ض�Ӧ��
		/// @return
		EXPORT_CLASS_API virtual void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence) = 0;

	};

}

#endif
