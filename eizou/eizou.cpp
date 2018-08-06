
#include <iostream>
#include <string>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>  

using namespace std;
using namespace cv;

	



void faceTest(int type, int num)
{
	String facefile = "haarcascade_frontalface_alt.xml";
	CascadeClassifier faceCascader;
	faceCascader.load(facefile);

	namedWindow("Camera", CV_WINDOW_AUTOSIZE);
	VideoCapture capture(0);
	Mat frame;
	Mat gray;
	vector<Rect> faces;
	int sn = 0;

	int ida[10]={0,0,0,0,0,0,0,0,0,0};
	int id = -1;
	String eyefile = "haarcascade_lefteye_2splits.xml";
	CascadeClassifier eyeCascader;
	eyeCascader.load(eyefile);
	Mat face;
	Ptr<FaceRecognizer> model;
	model = createLBPHFaceRecognizer();
	try {
		model->load("model.xml");
	}
	catch (Exception e2) {}
	while (capture.read(frame)) {
		
		const static Scalar colors[] = { CV_RGB(0,0,255),
			CV_RGB(0,128,255),
			CV_RGB(0,255,255),
			CV_RGB(0,255,0),
			CV_RGB(255,128,0),
			CV_RGB(255,255,0),
			CV_RGB(255,0,0),
			CV_RGB(255,0,255) };
		Mat gray, smallImg(cvRound(frame.rows / 2), cvRound(frame.cols / 2), CV_8UC1);
		cvtColor(frame, gray, CV_BGR2GRAY);
		cv::resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
		cv::equalizeHist(smallImg, smallImg);
		faceCascader.detectMultiScale(smallImg, faces,
			1.1, 2, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT
			//|CV_HAAR_DO_ROUGH_SEARCH
			| CV_HAAR_SCALE_IMAGE
			,
			Size(30, 30));

		for (size_t faceSize = 0;faceSize<faces.size();faceSize++)
		{

			Point p1, p2;
			Rect f;
			f.x = faces[static_cast<int>(faceSize)].x * 2;
			f.y = faces[static_cast<int>(faceSize)].y * 2;
			f.width = faces[static_cast<int>(faceSize)].width * 2;
			f.height = faces[static_cast<int>(faceSize)].height * 2;

			p1.x = faces[static_cast<int>(faceSize)].x*2;
			p1.y = faces[static_cast<int>(faceSize)].y*2;
			p2.x = (faces[static_cast<int>(faceSize)].x + faces[static_cast<int>(faceSize)].width)*2;
			p2.y = (faces[static_cast<int>(faceSize)].y + faces[static_cast<int>(faceSize)].height)*2;
			Scalar color = colors[0];
			rectangle(frame, p1, p2, color, 3, 8, 0);

			face = frame(f);
			cvtColor(face, gray, cv::COLOR_RGB2GRAY);
			equalizeHist(gray, gray);


			vector<Rect> objs;
			eyeCascader.detectMultiScale(gray, objs, 1.2, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
			int count = 0;

			for (vector<Rect>::const_iterator it = objs.begin(); it != objs.end(); ++it) {
				count++;
				rectangle(frame, Point(it->x + f.x, it->y + f.y),
					Point(it->x + f.x + it->width, it->y + f.y + it->height),
					Scalar(0, 0, 255), 2, CV_AA);
			}
			if (count >= 2) {
				cvtColor(face, gray, cv::COLOR_RGB2GRAY);
				equalizeHist(gray, gray);
				if (type == 2) {
					vector<Mat> images;
					vector<int> labels;
					images.push_back(gray);
					labels.push_back(num);
					model->update(images, labels);
				}
				else if (type == 3) {
					ida[model->predict(gray)]++;
					if (ida[model->predict(gray)] == 10)
					{
						id = model->predict(gray);
						for (int i = 0;i < 10;i++) {
							ida[i] = 0;
						}
					}
				}

			}
		}
		if (type==3) putText(frame, "Your number is: "+std::to_string(id), Point(100, 100), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0,255), 2);




		imshow("Camera", frame);
		char key = waitKey(50);
		if (key == 27) {
			model->save("model.xml");
			return;
		}
	}
}




int main()
{

	int i;
	cout << "1: Face recogonize" << endl;
	cout << "2: Record your face" << endl;
	cout << "3: Verify your face" << endl;
	cout << "9: Reset" << endl;
	cout << "0: Exit" << endl;
	cin >> i;
	while (i != 0) {
		if (i == 9) {
			Ptr<FaceRecognizer> newm;
			newm = createLBPHFaceRecognizer();
			newm->save("model.xml");
		}
		else {
			int num = -1;
			if (i == 2) {
				cout << "Input you number(1-9):" << endl;
				cin >> num;
				while (num < 1 || num>9) {
					cout << "Input you number again(1-9):" << endl;
					cin >> num;
				}
			}
			faceTest(i, num);
		}
		cout << "1: Face recogonize" << endl;
		cout << "2: Record your face" << endl;
		cout << "3: Verify your face" << endl;
		cout << "9: Reset" << endl;
		cout << "0: Exit" << endl;
		cin >> i;
	}

	return 0;
}