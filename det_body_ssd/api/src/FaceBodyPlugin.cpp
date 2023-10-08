#include "FaceBodyPlugin.h"
#include "base64.h"
#include <chrono>   
#include <iomanip>
#include "json/document.h"  
#include "json/prettywriter.h"
using namespace rapidjson;
using namespace std;
using namespace chrono;
#ifdef WIN32
#pragma comment( lib,"winmm.lib" )
#else
typedef unsigned long       DWORD;
#include <time.h> 
#endif
using namespace base::base64;


const int BUFFER_SIZE = 1024 * 1024 * 10;
FaceBodyPlugin::FaceBodyPlugin()
{
	m_pBuffer1 = new char[BUFFER_SIZE];
}


FaceBodyPlugin::~FaceBodyPlugin()
{
}



int FaceBodyPlugin::InvokeMethod(const char* strCommand,
	const std::map<std::string, std::string>& args,
	std::map<std::string, std::string>& out
)
{
	
	return -1;
}


int FaceBodyPlugin::InvokeMethod(const char* strCommand,
	const char* args,
	char* out
)
{
	if (strcmp("body.detect.single", strCommand) == 0)
	{
		try
		{
			
			Document document;
			document.Parse<0>(args);
			if (document.HasParseError() )
			{
				std::cout << "Json Parse error: "  << std::endl;
				return 1;
			}

			if (!document.HasMember("img"))
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}
			Value &img = document["img"];

			if (!img.IsString())
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}
			std::string imgbase64 = img.GetString();
			int len = calc_data_len(imgbase64.c_str(), imgbase64.length());

			//char * pOut = new char[len + 1];
			memset(m_pBuffer1, 0, BUFFER_SIZE);
			//auto start = system_clock::now();
			base64_decode(imgbase64.c_str(), imgbase64.length(), m_pBuffer1);
			//auto end = system_clock::now();
			//std::cout << "cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

			std::vector<char> data(m_pBuffer1, m_pBuffer1 + len);
			
			//delete[] pOut;
			cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);

			std::vector<cv::Rect> rectangles;
			std::vector<float> confidence;
			m_body_detector_server->detection(image, rectangles, confidence);


			
			Document jsondoc;
			jsondoc.SetObject();    //key-value �൱��map
								//doc.Setvalue();        //������ �൱��vector
			Document::AllocatorType &allocator = jsondoc.GetAllocator(); //��ȡ������ 

																		//����array
			Value rectanglesArray(rapidjson::kArrayType);//����һ��Array���͵�Ԫ��
			Value confidenceArray(rapidjson::kArrayType);//����һ��Array���͵�Ԫ��


			for (int i = 0; i < rectangles.size(); i++)
			{
				std::stringstream stream;

				int x = rectangles[i].x;
				int y = rectangles[i].y;
				int width = rectangles[i].width;
				int height = rectangles[i].height;
				rapidjson::Value object(rapidjson::kObjectType);
				object.AddMember("x", x, allocator);
				object.AddMember("y", y, allocator);
				object.AddMember("width", width, allocator);
				object.AddMember("height", height, allocator);
				rectanglesArray.PushBack(object, allocator);

				std::stringstream stream2;
				stream2 << confidence[i];
				std::string string_temp2 = stream2.str();
				confidenceArray.PushBack(rapidjson::Value((char*)string_temp2.c_str(), allocator), allocator);
			}
			jsondoc.AddMember("rectangles", rectanglesArray, allocator);    //�������
			jsondoc.AddMember("confidence", confidenceArray, allocator);    //�������
			StringBuffer buffer;
			Writer<StringBuffer> writer(buffer);
			jsondoc.Accept(writer);
			buffer.GetString();
			strcpy(out,buffer.GetString());
			return 0;
		}
		catch (std::exception& e)
		{
			std::cout << "Standard exception : " << e.what() << std::endl;
		}
		catch (...)
		{
			std::cout << "unknown exception : " << std::endl;
		}
	}
	else
	{
		std::cout << "no found function to call " << args << std::endl;
	}

	return -1;
}


int FaceBodyPlugin::GetInterface(
	const std::map<std::string, std::string>& args,
	std::map<std::string, std::string>& out
)
{
	std::cout << "FacePlugin::GetInterface" << std::endl;
	return 1;
}

