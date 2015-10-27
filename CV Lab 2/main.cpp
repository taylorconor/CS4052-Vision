#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

#define BW_THRESH_RATIO	0.7
#define LABEL_THRESH	25

/**
 finds the 'midpoint' (center point) of each bottle in the input matrix
 
 @param img input matrix
 
 @return the x coordinates of the midpoints of the bottles
 */
vector<int> getMidpoints(Mat img) {
	Mat gray, binary, top_binary;
	
	// perform an otsu threshold on the image to make it binary
	cvtColor(img, gray, CV_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY|THRESH_OTSU);
	
	// focus on the top 20% of the image (the tops of the bottle caps) since
	// each bottle is very easily discernable there
	Rect top = Rect(0, 0, binary.cols, binary.rows * 0.2);
	top_binary = binary(top);
	
	// perform kmeans with just one center, so that the 'centers' output matrix
	// is only one pixel in height
	int k = 1, attempts = 1;
	Mat labels, centers;
	top_binary.convertTo(top_binary, CV_32F);
	kmeans(top_binary, k, labels,
		   TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
		   attempts, KMEANS_PP_CENTERS, centers);
	
	// get the coordinates of the white pixels only. these are the tops of the
	// bottle caps
	centers.convertTo(centers, CV_8UC1);
	std::vector<cv::Point2i> locations;
	findNonZero(centers, locations);
	
	// figure out the centre point of each cluster of white pixels
	vector<int> midpoints;
	int prev = 0, start = -1;
	for (int i = 0; i < locations.size(); i++) {
		int p = locations[i].x;
		if (start == -1) {
			start = p;
		}
		// assumes that no two bottlecaps will be closer than 10 pixels together
		else if (p - prev > 10 || i == locations.size() - 1) {
			midpoints.push_back(((prev-start)/2)+start);
			start = p;
		}
		prev = p;
	}
	
	return midpoints;
}

/**
 finds bounding boxes for each bottle in an image of multiple bottles
 
 @param img input matrix
 @param midpoints the midpoint of each bottle in the input matrix
 
 @return a vector of bounding boxes that each fit around one glue bottle in
	the input matrix
 */
vector<Rect> getBounds(Mat img, vector<int> midpoints) {
	vector<Rect> bounds;
	
	// handle edge case where there's only one bottle in the input matrix
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
			bounds.push_back(Rect(prev_start, 0,
								  img.cols-prev_start, img.rows));
		}
	}
	return bounds;
}

/**
 gets the amount of white pixels with respect to the total amount of pixels in
 the thresholded input image, irrespective of luminance
 
 @param img input matrix
 
 @return ratio of white pixels to total pixels in the processed matrix
 */
double getRatio(Mat img) {
	// remove the top part of the image, leaving the bottom square, which
	// contains the face of the bottle
	Rect r = Rect(0, img.rows - img.cols, img.cols, img.cols);
	Mat crop = img(r);
	
	Mat hls, gray, binary;
	cvtColor(crop, hls, CV_BGR2HLS);
	
	// remove the luminance channel to enhance the difference between bottles
	// with labels and bottles without
	Mat channel[3];
	split(hls, channel);
	channel[0] = channel[1] = Mat::zeros(hls.rows, hls.cols, CV_8UC1);
	merge(channel, 3, hls);
	
	// convert back to binary (without luminance channel) and do a manual
	// threshold so the ratio of white pixels to total pixels can be calculated
	cvtColor(hls, gray, CV_BGR2GRAY);
	threshold(gray, binary, LABEL_THRESH, 255, THRESH_BINARY);
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
		// find the bounding box for each bottle in the image
		vector<Rect> bounds = getBounds(img, midpoints);
		vector<Rect> no_label;
		
		for (int j = 0; j < bounds.size(); j++) {
			// crop each bottle out of the image and find it's threshold ratio
			Mat crop = img(bounds[j]);
			double ratio = getRatio(crop);
			cout << "Image" << j << "," << i << " = " << ratio << endl;
			
			// if the threshold ratio falls below BW_THRESH_RATIO, then the
			// bottle has no label
			if (ratio < BW_THRESH_RATIO) {
				no_label.push_back(bounds[j]);
			}
		}
		
		
		// show the bottles with rectangles drawn around bottles with no labels
		for (int i = 0; i < no_label.size(); i++) {
			cout << no_label[i] << endl;
			rectangle(img, no_label[i], Scalar(0,0,255), 5, 8);
		}
		imshow(argv[i], img);
	}
	
	// quit program on keypress
	waitKey(0);
	return 0;
}
