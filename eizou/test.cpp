#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
using namespace std;
using namespace cv;
//人脸识别
void faceTest()
{
	String facefile = "G:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
	String leyefile = "G:\\opencv\\build\\etc\\haarcascades\\haarcascade_lefteye_2splits.xml";
	String reyefile = "G:\\opencv\\build\\etc\\haarcascades\\haarcascade_righteye_2splits.xml";

	CascadeClassifier faceCascader, leyeCascader, reyeCascader;
	faceCascader.load(facefile);
	leyeCascader.load(leyefile);
	reyeCascader.load(reyefile);

	
	namedWindow("摄像头", CV_WINDOW_AUTOSIZE);
	VideoCapture capture(0);//打开摄像头
	Mat frame;
	Mat gray;
	vector<Rect> faces;
	int sn = 0;
	//实时读取摄像头的图像帧
	while (capture.read(frame)) {
		//图像变灰
		cvtColor(frame, gray, COLOR_RGB2GRAY);
		equalizeHist(gray, gray);
		faceCascader.detectMultiScale(gray, faces, 1.2, 3, 0, Size(30, 30));
		for (size_t faceSize = 0;faceSize<faces.size();faceSize++)
		{
			Rect roi;
			roi.x = faces[static_cast<int>(faceSize)].x;
			roi.y = faces[static_cast<int>(faceSize)].y;
			roi.width = faces[static_cast<int>(faceSize)].width;
			roi.height = faces[static_cast<int>(faceSize)].height;
			Mat faceROI = frame(roi);
			//在人脸区域画一个矩形
			rectangle(frame, faces[static_cast<int>(faceSize)], Scalar(0, 0, 255), 2, 8, 0);
			sn++;
			//将sn整型值转为字符串
			stringstream stream; 
			stream << sn;
			//生产一个新的文件名
			String snStr = "G:\\test\\opencv\\img-" + stream.str() + ".jpg";
			cout << snStr << endl;
			//imwrite(snStr, faceROI);
		}
		imshow("摄像头", frame);
		//必须加时延，否则无法显示图像
		char key = waitKey(50);
		//按ESC键退出
		if (key == 27) {
			break;
		}
	}
}
int main()
{
	faceTest();
	waitKey(0);
	return 0;
}