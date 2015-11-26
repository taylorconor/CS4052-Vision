#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include "Histograms.cpp"

using namespace cv;
using namespace std;

#define BOOKIMG		"BookView"
#define BOOKAMT		25
#define PAGEIMG		"Page"
#define PAGEAMT		13

/**
 find the coordinates of the four corners of a noiseless binary image
 
 @param img noiseless binary input image
 
 @return a vector of Points of the corner coordinates (in no particular order)
 */
vector<Point> findCornerPoints(Mat img) {
	vector<Point> v;
	Point right, bottom, top;
	int startx = -1, endx = -1, starty = -1, endy = -1;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			uchar pval = img.at<uchar>(j,i);
			if (pval > 0) {
				if (startx == -1) {
					v.push_back(Point(i,j));
					startx = i;
				}
				if (starty == -1 || starty > j) {
					top = Point(i,j);
					starty = j;
				}
				if (j > endy) {
					endy = j;
					bottom = Point(i,j);
				}
				if (i > endx) {
					endx = i;
					right = Point(i,j);
				}
			}
		}
	}
	v.push_back(bottom);
	v.push_back(right);
	v.push_back(top);
	return v;
}

// perform a simple closing
Mat closing(Mat img, int amt=1) {
	Mat tmp = img.clone();
	for (int i = 0; i < amt; i++) {
		dilate(tmp, tmp, Mat());
		erode(tmp, tmp, Mat());
	}
	return tmp;
}

// perform a simple erosion
Mat erosion(Mat img, int amt=1) {
	for (int i = 0; i < amt; i++)
		erode(img, img, Mat());
	return img;
}

// perform a simple dilation
Mat dilate(Mat img, int amt=1) {
	for (int i = 0; i < amt; i++)
		dilate(img, img, Mat());
	return img;
}

int main(int argc, char* argv[]) {
	
	if (argc < 1) {
		cout << "Usage: " << argv[0] << " [img dir]" << endl;
		return 0;
	}
	
	string dir = argv[1];
	Mat bluePixels = imread(dir+"/BlueBookPixelsNew.png");
	
	// calculate histogram of blue pixels for back projection
	cvtColor(bluePixels, bluePixels, CV_BGR2HLS);
	ColourHistogram h = ColourHistogram(bluePixels, 4);
	
	for (int i = 1; i <= BOOKAMT; i++) {
		string s = dir+"/"+BOOKIMG+to_string(i)+".jpg";
		Mat img = imread(s);
		
		// blow up the image 4x to make back projection calculations more
		// effective
		resize(img, img, Size(), 4, 4);
		Mat binary, backProject, eroded, dilated, hls, mask, masked;
		cvtColor(img, hls, CV_BGR2HLS);
		
		// build a mask to remove everything that's not part of the page. this
		// is most effectively achieved by thresholding the red channel and
		// performing a series of closings, followed my erosions to remove noise
		vector<Mat> spl;
		split(img, spl);
		threshold(spl[0], binary, 0, 255, THRESH_BINARY|THRESH_OTSU);
		binary = closing(binary, 3);
		mask = erosion(binary, 3);
		
		// apply the mask to the image
		img.copyTo(masked, mask);
		cvtColor(masked, masked, CV_BGR2HLS);
		
		// back project blue pixels
		backProject = dilate(h.BackProject(masked), 3);
		
		// reduce image back to original size
		resize(backProject, backProject, Size(), 0.25, 0.25);

		// find the four corner points in the back projected image
		vector<Point> corners = findCornerPoints(backProject);
		cvtColor(backProject, backProject, CV_GRAY2BGR);
		for (int i = 0; i < corners.size(); i++) {
			circle(backProject, corners[i], 3, Scalar(0,0,255));
		}
		
		imshow(s, backProject);
		
		waitKey(0);
	}
	
	// quit program on keypress
	waitKey(0);
	return 0;
}
