#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

#define BOOKIMG		"BookView"
#define BOOKAMT		25
#define PAGEIMG		"Page"
#define PAGEAMT		13

Mat easyCalcHist(Mat img) {
	Mat hist;
	int bbins = 8, gbins = 8, rbins = 12;
	int histSize[] = {bbins, gbins, rbins};
	float branges[] = {0, 180}, granges[] = {0, 180}, rranges[] = {0, 255};
	const float* ranges[] = {branges, granges, rranges};
	int channels[] = {0, 1, 2};
	
	calcHist(&img, 1, channels, Mat(), hist, 3, histSize, ranges);
	return hist;
}

Mat easyCalcBackProject(Mat img, Mat hist) {
	Mat backProject;
	float branges[] = {0, 255}, granges[] = {0, 255}, rranges[] = {0, 255};
	const float* ranges[] = {branges, granges, rranges};
	int channels[] = {0, 1, 2};
	
	calcBackProject(&img, 1, channels, hist, backProject, ranges);
	return backProject;
}

int main(int argc, char* argv[]) {
	
	if (argc < 1) {
		cout << "Usage: " << argv[0] << " [img dir]" << endl;
		return 0;
	}
	
	string dir = argv[1];
	Mat bluePixels = imread(dir+"/BlueBookPixels.png");
	Mat blueHist = easyCalcHist(bluePixels);
	
	for (int i = 1; i <= BOOKAMT; i++) {
		string s = dir+"/"+BOOKIMG+to_string(i)+".jpg";
		Mat img = imread(s);
		Mat binary, backProject, dilated;
		
		backProject = easyCalcBackProject(img, blueHist);
		threshold(backProject, binary, 0, 255, THRESH_BINARY|THRESH_OTSU);
		dilate(binary, dilated, Mat());
		
		imshow(s, binary);
		waitKey(0);
	}
	
	// quit program on keypress
	waitKey(0);
	return 0;
}
