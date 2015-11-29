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

// represents and identifies the corners found in the book image
struct Corners {
	Point top_left;
	Point top_right;
	Point bottom_left;
	Point bottom_right;
	
	vector<Point2f> toVector() {
		vector<Point2f> p;
		p.push_back(Point(top_left.x, top_left.y-2));
		p.push_back(Point(top_right.x+5, top_right.y));
		p.push_back(Point(bottom_right.x+5, bottom_right.y+5));
		p.push_back(Point(bottom_left.x-5, bottom_left.y+2));
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

// transform a book image to a rectangle using the provided corner points
Mat transformToRectangle(Mat img, Corners c) {
	cv::Mat result = cv::Mat::zeros(PAGEHEIGHT, PAGEWIDTH, CV_8UC3);
	
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(result.cols, 0));
	quad_pts.push_back(cv::Point2f(result.cols, result.rows));
	quad_pts.push_back(cv::Point2f(0, result.rows));
	
	// transformation matrix
	cv::Mat transmtx = cv::getPerspectiveTransform(c.toVector(), quad_pts);
	
	// apply perspective transformation
	cv::warpPerspective(img, result, transmtx, result.size());
	return result;
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

// convert input image to book image
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
	cvtColor(backProject, backProject, CV_GRAY2BGR);
	circle(backProject, corners.bottom_left, 3, Scalar(255, 255, 255), 10);
	circle(backProject, corners.top_left, 3, Scalar(0, 0, 255), 10);
	circle(backProject, corners.top_right, 3, Scalar(0, 255, 0), 10);
	circle(backProject, corners.bottom_right, 3, Scalar(255, 0, 0), 10);
	
	return transformToRectangle(img, corners);
}

// returns a list of all of the template images, paired with an edge image
// version so we don't have to compute the edges each time we try a match
vector<pair<Mat, Mat>> getTemplateImages(string dir, ColourHistogram h) {
	vector<pair<Mat, Mat>> v;
	for (int i = 1; i <= PAGEAMT; i++) {
		string s = dir+"/"+PAGEIMG+to_string(i)+".JPG";
		Mat img = imread(s);
		resize(img, img, Size(PAGEWIDTH, PAGEHEIGHT));
		
		pair<Mat, Mat> p;
		GaussianBlur(img, img, Size(3,3), 10);
		p.first = img;

		Mat edge;
		Canny(img, edge, 100, 200);
		// crop out the blue points/lines
		Rect r = Rect(20, 20, edge.cols - 40, edge.rows - 40);
		p.second = edge(r);
		
		v.push_back(p);
	}
	return v;
}

// find the index of the template image that matches the input image
int getMatchingImage(Mat img, vector<pair<Mat, Mat>>& templates) {
	int result = 0;
	double global_max_correlation = -1;
	
	Mat edge;
	cvtColor(img, img, CV_BGR2GRAY);
	Canny(img, edge, 50, 15);
	
	Mat correlation_img, test;
	double min_correlation, max_correlation;
	
	for (int i = 0; i < templates.size(); i++) {
		matchTemplate(edge, templates[i].second,
					  correlation_img, CV_TM_CCORR_NORMED);
		minMaxLoc(correlation_img, &min_correlation, &max_correlation);
		if (max_correlation > global_max_correlation) {
			global_max_correlation = max_correlation;
			result = i;
		}
	}
	imshow("src", edge);
	imshow("template", templates[result].second);
	return result;
}

// returns the two input images displayed side by side (for display only)
Mat getDisplayImage(Mat img, int imgno, Mat t, int tempno) {
	Size s1 = img.size();
	Size s2 = t.size();
	Mat result(s1.height, s1.width+s2.width, CV_8UC3);
	Mat left(result, Rect(0, 0, s1.width, s1.height));
	img.copyTo(left);
	Mat right(result, Rect(s1.width, 0, s2.width, s2.height));
	t.copyTo(right);
	
	putText(result, BOOKIMG+to_string(imgno), cvPoint(10,20),
			FONT_HERSHEY_PLAIN, 1, cvScalar(0,0,250), 1, CV_8UC3);
	putText(result, PAGEIMG+to_string(tempno), cvPoint(PAGEWIDTH+10,20),
			FONT_HERSHEY_PLAIN, 1, cvScalar(0,0,250), 1, CV_8UC3);
	
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
	
	vector<pair<Mat, Mat>> templates = getTemplateImages(dir, h);
	
	for (int i = 1; i <= BOOKAMT; i++) {
		string s = dir+"/"+BOOKIMG+to_string(i)+".jpg";
		Mat img = imread(s);
		
		Mat transformed = processImageToPage(img, h);
		int match = getMatchingImage(transformed, templates);
		
		Mat display = getDisplayImage(transformed, i,
									  templates[match].first, match);
		// show the two images side by side
		imshow(s, display);
		waitKey(0);
	}
	waitKey(0);
	return 0;
}
