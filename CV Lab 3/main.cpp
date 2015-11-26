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

Mat easyCalcHist(Mat img) {
	Mat hist;
	int bins = 4;
	int histSize[] = {bins, bins, bins};
	float r[] = {0, 255};
	const float* ranges[] = {r, r, r};
	int channels[] = {0, 1, 2};
	
	calcHist(&img, 1, channels, Mat(), hist, 3, histSize, ranges);
	
	return hist;
}

Mat easyCalcBackProject(Mat img, Mat hist) {
	Mat backProject;
	float branges[] = {0, 255}, granges[] = {0, 255}, rranges[] = {0, 255};
	const float* ranges[] = {branges, granges, rranges};
	int channels[] = {0, 1, 2};
	
	calcBackProject(&img, 1, channels, hist, backProject, ranges, 255);
	return backProject;
}

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

Mat findBlueDots(Mat img) {
	Vec3b black = {0,0,0};
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			Vec3b pixel = img.at<Vec3b>(Point(i,j));
			if (pixel[0] < (pixel[1]+pixel[2])*0.85) {
				img.at<Vec3b>(Point(i,j)) = black;
			}
		}
	}
	return img;
}

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

/*int main(int argc, char* argv[]) {
	
	if (argc < 1) {
		cout << "Usage: " << argv[0] << " [img dir]" << endl;
		return 0;
	}
	
	string dir = argv[1];
	Mat bluePixels = imread(dir+"/BlueBookPixelsNew2.png");
	cvtColor(bluePixels, bluePixels, CV_BGR2HLS);
	Mat blueHist = easyCalcHist(bluePixels);
	
	for (int i = 1; i <= BOOKAMT; i++) {
		string s = dir+"/"+BOOKIMG+to_string(i)+".jpg";
		Mat img = imread(s);
		Mat binary, backProject, eroded, gray, hls;
		cvtColor(img, hls, CV_BGR2HLS);
		cvtColor(img, gray, CV_BGR2GRAY);
		
		// find the minimum bounding box of the book (to make back projection
		// less noise-prone).
		threshold(gray, binary, 165, 255, THRESH_BINARY);
		erode(binary, eroded, Mat());
		erode(eroded, eroded, Mat());
		//CvRect crop = cropBlackBorder(eroded);
		
		backProject = easyCalcBackProject(hls, blueHist);
		threshold(backProject, backProject, 10, 255, THRESH_BINARY);
		
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
}*/

int main(int argc, char* argv[]) {
	
	if (argc < 1) {
		cout << "Usage: " << argv[0] << " [img dir]" << endl;
		return 0;
	}
	
	string dir = argv[1];
	Mat bluePixels = imread(dir+"/BlueBookPixelsNew.png");
	cvtColor(bluePixels, bluePixels, CV_BGR2HLS);
	ColourHistogram h = ColourHistogram(bluePixels, 5);
	
	for (int i = 1; i <= BOOKAMT; i++) {
		string s = dir+"/"+BOOKIMG+to_string(i)+".jpg";
		Mat img = imread(s);
		Mat binary, backProject, eroded, gray, hls;
		cvtColor(img, hls, CV_BGR2HLS);
		cvtColor(img, gray, CV_BGR2GRAY);
		
		// find the minimum bounding box of the book (to make back projection
		// less noise-prone).
		threshold(gray, binary, 165, 255, THRESH_BINARY);
		erode(binary, eroded, Mat());
		erode(eroded, eroded, Mat());
		CvRect crop = cropBlackBorder(eroded);
		Mat cropped = hls(crop);
		
		backProject = h.BackProject(cropped);
		imshow(s, backProject);
		
		waitKey(0);
	}
	
	// quit program on keypress
	waitKey(0);
	return 0;
}
