#include <stdio.h>
#include <string>

#include "opencv\cv.h"
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "RBFNetwork.h"

using namespace std;
using namespace cv;

string data_dir = "D:/Osama/MyDrive/[Fourth Year][CS15]/2nd Semester/[NeuralNetworks15]/Final Project/Head Orientation Data set/Head Orientation Data set/";

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
	Mat image_gray_equalized;
	equalizeHist( image_gray, image_gray_equalized );
	datapoint data_point(image_gray_equalized.rows*image_gray_equalized.cols);
	int i=0;
	MatConstIterator_<uchar> it = image_gray_equalized.begin<uchar>(), it_end = image_gray_equalized.end<uchar>();
	for(; it != it_end; ++it)
		data_point[i++] = ((double)(*it)/128.0)-1.0;
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
	std::srand ( unsigned ( std::time(0) ) );
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
	loadTrainingData(testing_data,test_labels, 30);

	RBFNetwork RBFNN(training_data,training_labels,3);

	for(int rbf_units = 5 ; rbf_units<=20 ; rbf_units++)
	{
		for(double learning = 0.01 ; learning<=2.0 ; learning*=2.0)
		{
			printf("RBF Network with %d units, learning rate=%f\n", rbf_units, learning);
			double mse=0;
			double acc  = RBFNN.startTraining(rbf_units, learning, 100, mse, true);
			//printf(" Acc=%.6f , MSE=%.6f\n", acc, mse);
		}
	}
	

	return 0;
}