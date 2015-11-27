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

#define PAGEWIDTH	350
#define PAGEHEIGHT	513

struct Corners {
	Point top_left;
	Point top_right;
	Point bottom_left;
	Point bottom_right;
	
	vector<Point2f> toVector() {
		vector<Point2f> p;
		p.push_back(top_left);
		p.push_back(top_right);
		p.push_back(bottom_right);
		p.push_back(bottom_left);
		return p;
	}
};

/**
 find the coordinates of the four corners of a noiseless binary image
 
 @param img noiseless binary input image
 
 @return a Corners object identifying the coordinates of each corner
 */
Corners findCornerPoints(Mat img) {
	Corners c;
	Point right, bottom, top;
	int startx = -1, endx = -1, starty = -1, endy = -1;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			uchar pval = img.at<uchar>(j,i);
			if (pval > 0) {
				if (startx == -1) {
					c.bottom_left = Point(i,j);
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
	c.bottom_right = bottom;
	c.top_right = right;
	c.top_left = top;
	return c;
}

Mat transformToRectangle(Mat img, Corners c) {
	// Define the destination image
	cv::Mat quad = cv::Mat::zeros(PAGEHEIGHT, PAGEWIDTH, CV_8UC3);
	
	// Corners of the destination image
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
	quad_pts.push_back(cv::Point2f(0, quad.rows));
	
	// Get transformation matrix
	cv::Mat transmtx = cv::getPerspectiveTransform(c.toVector(), quad_pts);
	
	// Apply perspective transformation
	cv::warpPerspective(img, quad, transmtx, quad.size());
	return quad;
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

Mat processImageToPage(Mat img, ColourHistogram h) {
	// blow up the image 4x to make back projection calculations more
	// effective
	resize(img, img, Size(), 4, 4);
	Mat binary, backProject, eroded, dilated, hls, mask, masked, transformed;
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
	resize(img, img, Size(), 0.25, 0.25);
	resize(backProject, backProject, Size(), 0.25, 0.25);
	
	// find the four corner points in the back projected image
	Corners corners = findCornerPoints(backProject);
	/*cvtColor(backProject, backProject, CV_GRAY2BGR);
	circle(backProject, corners.bottom_left, 3, Scalar(255, 255, 255));
	circle(backProject, corners.top_left, 3, Scalar(0, 0, 255));
	circle(backProject, corners.top_right, 3, Scalar(0, 255, 0));
	circle(backProject, corners.bottom_right, 3, Scalar(255, 0, 0));*/
	
	return transformToRectangle(img, corners);
}

vector<Mat> getTemplateImages(string dir, ColourHistogram h) {
	vector<Mat> v;
	for (int i = 1; i <= PAGEAMT; i++) {
		string s = dir+"/"+PAGEIMG+to_string(i)+".JPG";
		Mat img = imread(s);
		resize(img, img, Size(PAGEWIDTH, PAGEHEIGHT));
		img.convertTo(img, CV_32F);
		v.push_back(img);
	}
	return v;
}

int getMatchingImage(Mat img, vector<Mat>& templates) {
	int result = 0;
	double maxCorrelation = -1;
	img.convertTo(img, CV_32F);
	
	for (int i = 0; i < templates.size(); i++) {
		double c = abs(compareHist(img, templates[i], CV_COMP_INTERSECT));
		cout << "coeff for " << i << ": " << c << endl;
		if (c > maxCorrelation) {
			maxCorrelation = c;
			result = i;
		}
	}
	return result;
}

Mat getDisplayImage(Mat img, int imgno, Mat t, int tempno) {
	t.convertTo(t, CV_8U);
	Size s1 = img.size();
	Size s2 = t.size();
	Mat result(s1.height, s1.width+s2.width, CV_8UC3);
	Mat left(result, Rect(0, 0, s1.width, s1.height));
	img.copyTo(left);
	Mat right(result, Rect(s1.width, 0, s2.width, s2.height));
	t.copyTo(right);
	
	putText(result, BOOKIMG+to_string(imgno), cvPoint(10,20),
			FONT_HERSHEY_PLAIN, 1, cvScalar(0,0,250), 1, CV_AA);
	putText(result, PAGEIMG+to_string(tempno), cvPoint(PAGEWIDTH+10,20),
			FONT_HERSHEY_PLAIN, 1, cvScalar(0,0,250), 1, CV_AA);
	
	return result;
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
	
	vector<Mat> templates = getTemplateImages(dir, h);
	
	for (int i = 1; i <= BOOKAMT; i++) {
		string s = dir+"/"+BOOKIMG+to_string(i)+".jpg";
		Mat img = imread(s);
		
		Mat transformed = processImageToPage(img, h);
		int match = getMatchingImage(transformed, templates);
		
		Mat display = getDisplayImage(transformed, i, templates[match], match);
		imshow(s, display);
		
		waitKey(0);
	}
	
	// quit program on keypress
	waitKey(0);
	return 0;
}
