#include <stdio.h>
#include <string>
#include <algorithm>
#include <time.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"

#include "RBFNetwork.h"

using namespace std;
using namespace cv;

string data_dir = "D:/Osama/MyDrive/[Fourth Year][CS15]/2nd Semester/[NeuralNetworks15]/Final Project/Head Orientation Data set/Head Orientation Data set/";
string label_result[]= { "Facing Front.", "Facing Left.", "Facing Right." };

string intToStr(int num)
{
	if(num==0)return "0";
	string ret = "";
	while(num>0)
	{
		ret += (num%10)+'0';
		num/=10;
	}
	reverse(ret.begin(),ret.end());
	return ret;
}

datapoint imageToDoubleVec(cv::Mat image)
{
	Mat image_gray;
	cvtColor(image,image_gray,CV_RGB2GRAY);
	equalizeHist( image_gray, image_gray );
	datapoint data_point(image_gray.rows*image_gray.cols);
	int i=0;
	MatConstIterator_<uchar> it = image_gray.begin<uchar>(), it_end = image_gray.end<uchar>();
	for(; it != it_end; ++it)
		data_point[i++] = ((double)(*it)/128.0)-1;
	return data_point;
}

__inline void loadTrainingData(vector<datapoint> &training_data, vector<int> &training_labels, int training_size)
{
	string training_dir = data_dir+"Training/";
	// Front
	for(int i = 0 ; i<training_size ; i++)
	{
		Mat image = imread(training_dir+"Data_Front/head (" + intToStr(i+1) + ").png");
		if(!image.rows)
		{
			printf("Error Loading Image!\n");
			continue;
		}
		training_data.push_back(imageToDoubleVec(image));
		training_labels.push_back(0);
	}

	// Left
	for(int i = 0 ; i<training_size ; i++)
	{
		Mat image = imread(training_dir+"Data_Left/head (" + intToStr(i+1) + ").png");
		if(!image.rows)
		{
			printf("Error Loading Image!\n");
			continue;
		}
		training_data.push_back(imageToDoubleVec(image));
		training_labels.push_back(1);
	}

	// Right
	for(int i = 0 ; i<training_size ; i++)
	{
		Mat image = imread(training_dir+"Data_Right/head (" + intToStr(i+1) + ").png");
		if(!image.rows)
		{
			printf("Error Loading Image!\n");
			continue;
		}
		training_data.push_back(imageToDoubleVec(image));
		training_labels.push_back(2);
	}
	vector<int>perm(training_data.size());
	vector<datapoint> temp = training_data;
	vector<int> temp2 = training_labels;
	for(int i =0 ;i<training_data.size() ; i++)	perm[i]=i;
	srand (unsigned(time(0)));
	random_shuffle(perm.begin(),perm.end());
	for(int i =0 ;i<training_data.size() ; i++) training_data[perm[i]] = temp[i],training_labels[perm[i]] = temp2[i];
}

__inline void loadTestingData(vector<datapoint> &testing_data, vector<int> &testing_labels, int testing_size)
{
	string testing_dir = data_dir+"Testing/";
	// Front
	for(int i = 0 ; i<testing_size ; i++)
	{
		Mat image = imread(testing_dir+"Test_Front/head (" + intToStr(i+1) + ").png");
		if(!image.rows)
		{
			printf("Error Loading Image!\n");
			continue;
		}
		testing_data.push_back(imageToDoubleVec(image));
		testing_labels.push_back(0);
	}

	// Left
	for(int i = 0 ; i<testing_size ; i++)
	{
		Mat image = imread(testing_dir+"Test_Left/head (" + intToStr(i+1) + ").png");
		if(!image.rows)
		{
			printf("Error Loading Image!\n");
			continue;
		}
		testing_data.push_back(imageToDoubleVec(image));
		testing_labels.push_back(1);
	}

	// Right
	for(int i = 0 ; i<testing_size ; i++)
	{
		Mat image = imread(testing_dir+"Test_Right/head (" + intToStr(i+1) + ").png");
		if(!image.rows)
		{
			printf("Error Loading Image!\n");
			continue;
		}
		testing_data.push_back(imageToDoubleVec(image));
		testing_labels.push_back(2);
	}
}



int main()
{
	vector<datapoint> training_data, testing_data;
	vector<int> training_labels, test_labels;
	loadTrainingData(training_data,training_labels, 50);
	loadTestingData(testing_data,test_labels, 30);

	RBFNetwork RBFNN(training_data,training_labels,3);
	double mse=0;
	RBFNN.startTraining(5, 0.01, 10, mse, true);

	printf("Choose: \n");
	printf("1) Real-time Testing\n");
	printf("2) Testing on a test data set\n");
	int choice; scanf("%d", &choice);

	if(choice==1)
	{
		VideoCapture capture;
		Mat frame;
		CascadeClassifier profilefaceCascade;
		if( !profilefaceCascade.load( "C:/opencv/sources/data/haarcascades/HS22x20/HS.xml" ) )
			printf("--(!)Error loading\n");

		string window_name = "Capture - Eye Iris detection";
		RNG rng(12345);
		capture.open( 0 );
		if( capture.isOpened() )
		{
			while(true)
			{
				capture >> frame;
				vector<Rect> profilefaceRect;
				if( !frame.empty() )  
				{
					profilefaceCascade.detectMultiScale(frame, profilefaceRect, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE);
					if(profilefaceRect.size())
					{
						Size size(50,50);//the dst image size
						Mat cropedImage = frame(profilefaceRect[0]);
						Mat resizedImage;
						resize(cropedImage,resizedImage,size);
						cvtColor(cropedImage,cropedImage,CV_RGB2GRAY);
						equalizeHist( cropedImage, cropedImage );
						datapoint data_point = imageToDoubleVec(resizedImage);
						double err=0;
						int predict = RBFNN.predictLabel(data_point, err);
						rectangle(frame,profilefaceRect[0],Scalar(255,0,0));
						putText(frame, label_result[predict], cvPoint(30,30), 
							FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
					}
					imshow( window_name, frame );
				}
				else { printf(" --(!) No captured frame -- Break!"); break; }
				int c = waitKey(10);
				if( (char)c == 'c') { break; }
				if( (char)c == 's')
				{
					imwrite( "test.jpg", frame );
				}


			}
		}
	}
	else if(choice==2)
	{
		RBFNN.startTesting(testing_data,test_labels);
	}

	//// Experimenting results for different number of RBF units and learning rates
	//for(int rbf_units = 5 ; rbf_units<=100 ; rbf_units+=2)
	//{
	//	for(double learning = 0.01 ; learning<=10.0 ; learning*=2.0)
	//	{
	//		printf("RBF Network with %d units, learning rate=%f\n", rbf_units, learning);
	//		double mse=0;
	//		double acc  = RBFNN.startTraining(rbf_units, learning, 100, mse, true);
	//		RBFNN.startTesting(testing_data,test_labels);
	//		//printf(" Acc=%.6f , MSE=%.6f\n", acc, mse);
	//	}
	//}


	return 0;
}