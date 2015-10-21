#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

vector<int> getMidpoints(Mat img) {
	Mat gray, binary, top_binary;
	cvtColor(img, gray, CV_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY|THRESH_OTSU);
	Rect top = Rect(0, 0, binary.cols, binary.rows * 0.2);
	top_binary = binary(top);
	
	int k = 1, attempts = 1;
	Mat labels, centers;
	top_binary.convertTo(top_binary, CV_32F);
	
	kmeans(top_binary, k, labels,
		   TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
		   attempts, KMEANS_PP_CENTERS, centers);
	
	centers.convertTo(centers, CV_8UC1);
	std::vector<cv::Point2i> locations;
	findNonZero(centers, locations);
	
	vector<int> midpoints;
	int prev = 0, start = -1;
	for (int i = 0; i < locations.size(); i++) {
		int p = locations[i].x;
		if (start == -1) {
			start = p;
		}
		else if (p - prev > 10 || i == locations.size() - 1) {
			midpoints.push_back(((prev-start)/2)+start);
			start = p;
		}
		prev = p;
	}
	
	return midpoints;
}

vector<Rect> getBounds(Mat img, vector<int> midpoints) {
	vector<Rect> bounds;
	
	// handle edge case where there's only one midpoint
	if (midpoints.size() == 1) {
		bounds.push_back(Rect(0, 0, img.cols, img.rows));
		return bounds;
	}
	
	int prev_start = 0;
	for (int i = 1; i < midpoints.size(); i++) {
		// get the rect around the previous midpoint
		Rect r = Rect(prev_start, 0,
					  ((midpoints[i-1]+midpoints[i])/2)-prev_start, img.rows);
		bounds.push_back(r);
		prev_start = (midpoints[i-1]+midpoints[i])/2;
		
		// handle the last midpoint in the image
		if (i == midpoints.size() - 1) {
			bounds.push_back(Rect(prev_start, 0, img.cols-prev_start, img.rows));
		}
	}
	return bounds;
}

double getRatio(Mat img) {
	Rect r = Rect(0, img.rows - img.cols, img.cols, img.cols);
	Mat crop = img(r);
	
	Mat hls, gray, binary;
	cvtColor(crop, hls, CV_BGR2HLS);
	
	Mat channel[3];
	split(hls, channel);
	channel[1] = Mat::zeros(hls.rows, hls.cols, CV_8UC1);
	merge(channel, 3, hls);
	
	cvtColor(hls, gray, CV_BGR2GRAY);
	threshold(gray, binary, 25, 255, THRESH_BINARY);
	
	double ratio = ((double)countNonZero(binary)/(double)binary.total())*100;

	return ratio;
}

int main(int argc, char* argv[]) {
	
	if (argc < 1) {
		cout << "Usage: " << argv[0]
			<< " [image 1] [image 2] [image n]" << endl;
	}

	for (int i = 1; i < argc; i++) {
		Mat img = imread(argv[i]);
		vector<int> midpoints = getMidpoints(img);
		vector<Rect> bounds = getBounds(img, midpoints);
		vector<Rect> no_label;
		
		for (int j = 0; j < bounds.size(); j++) {
			Mat crop = img(bounds[j]);
			
			double ratio = getRatio(crop);
			cout << "Image" << j << "," << i << " = " << ratio << endl;
			
			if (ratio < 1) {
				no_label.push_back(bounds[j]);
			}
		}
		
		
		// draw rectangles around bottles with no labels
		for (int i = 0; i < no_label.size(); i++) {
			cout << no_label[i] << endl;
			rectangle(img, no_label[i], Scalar(0,0,255), 5, 8);
		}
		
		imshow(argv[i], img);
	}
	
	waitKey(0);
	return 0;
}
