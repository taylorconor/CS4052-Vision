#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

#include "Video.cpp"

using namespace cv;
using namespace std;


CvRect cropBlackBorder(Mat img) {
	int startx = -1, endx = -1, starty = -1, endy = -1;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			uchar pval = img.at<uchar>(j,i);
			if (pval > 0) {
				if (startx == -1)
					startx = i;
				if (starty == -1 || starty > j)
					starty = j;
				if (j > endy)
					endy = j;
				if (i > endx)
					endx = i;
			}
		}
	}
	
	// add 10px padding on each side, if possible
	if (startx >= 10)
		startx -= 10;
	if (starty >= 10)
		starty -= 10;
	if (endx <= img.cols-10)
		endx += 10;
	if (endy <= img.rows-10)
		endy += 10;
	
	CvRect rect(startx, starty, endx-startx, endy-starty);
	return rect;
}

// remove noise that's not part of the bag
Mat cleanNoise(Mat img) {
	Mat res;
	erode(img, res, Mat());
	erode(res, res, Mat());
	dilate(res, res, Mat());
	return res;
}

int main(int argc, char* argv[]) {
	VideoCapture cap("/Users/Conor/Documents/College/CS4053/labs/labs/CV Lab 4/video/ObjectAbandonmentAndRemoval1.avi");
	if(!cap.isOpened())
		return -1;
	
	Mat edges;
	namedWindow("Video",1);
	namedWindow("Median",1);
	Mat frame, mbframe1, mbframe2, diff, tdiff;
	cap.read(frame);
	Rect r, finalr;
	bool isRectValid = false, isRectFinal = false;
	int finalCount = 0;
	MedianBackground medianBackground1(frame, 1.009, 4);
	MedianBackground medianBackground2(frame, 1.005, 4);
	while(cap.read(frame)) {
		// get current median frame and update based on current frame
		mbframe1 = medianBackground1.GetBackgroundImage();
		mbframe2 = medianBackground2.GetBackgroundImage();
		medianBackground1.UpdateBackground(frame);
		medianBackground2.UpdateBackground(frame);
		
		// get the absolute difference of the two different background models
		absdiff(mbframe1, mbframe2, diff);
		
		cvtColor(diff, diff, CV_BGR2GRAY);
		threshold(diff, tdiff, 50, 255, THRESH_BINARY);
		
		// try to clean some of the noise not related to the moving obejct
		Mat total_diff = cleanNoise(tdiff);

		if (countNonZero(total_diff) > 0) {
			Rect newr = cropBlackBorder(total_diff);
			if (!isRectValid) {
				// begin rectangle size tracking
				isRectValid = true;
				r = newr;
			} else {
				// keep growing the rectangle until it starts shrinking, then
				// maintain the max size it held
				if (newr.area() > r.area()) {
					r = newr;
				} else {
					finalr = r;
					isRectFinal = true;
				}
			}
		} else {
			// reset rectangle size tracking
			isRectValid = false;
		}
		
		// display the tracked rectangle for 40 frames
		if (isRectFinal && finalCount < 40) {
			rectangle(frame, finalr, Scalar(0,0,255), 4);
			finalCount++;
		} else {
			isRectFinal = false;
			finalCount = 0;
		}
		
		imshow("Video", frame);
		imshow("Median", total_diff);
		waitKey(1);
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
