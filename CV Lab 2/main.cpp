#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat getGrayscale(Mat img) {
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	return gray;
}

int main(int argc, char* argv[]) {
	
	if (argc < 1) {
		cout << "Usage: " << argv[0]
			<< " [image 1] [image 2] [image n]" << endl;
	}
		
	for (int i = 1; i < argc; i++) {
		Mat img = imread(argv[i]);
		Mat greyImg = getGrayscale(img);
		
		stringstream sstream;
		sstream << "Image " << i;
		imshow(sstream.str(), greyImg);
	}
	
	waitKey(0);
	return 0;
}
