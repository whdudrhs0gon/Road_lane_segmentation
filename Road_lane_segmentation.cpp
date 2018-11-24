#include "cv.hpp"
#include <iostream>


using namespace cv;
using namespace std;


int main() {
	VideoCapture cap("Road_image.mp4");
	Mat frame, line_white, line_yellow, mask_white, mask_yellow, white, yellow, foreground, foreground1, foreground2, result;

	Point p[1][4];
	p[0][0] = Point(200, 380);
	p[0][1] = Point(420, 260);
	p[0][2] = Point(450, 280);
	p[0][3] = Point(370, 390);

	Point q[1][4];
	q[0][0] = Point(480, 280);
	q[0][1] = Point(700, 390);
	q[0][2] = Point(580, 390);
	q[0][3] = Point(450, 280);

	Point r[1][4];
	r[0][1] = Point(400, 300);
	r[0][0] = Point(270, 390);
	r[0][2] = Point(455, 280);
	r[0][3] = Point(685, 390);

	const Point* ppt[1] = { p[0] };
	const Point* qqt[1] = { q[0] };
	const Point* rrt[1] = { r[0] };
	int npt[] = { 4 };
	int nqt[] = { 4 };
	int nrt[] = { 4 };

	Mat bg_yellow(Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), CV_8UC1, Scalar(0));
	fillPoly(bg_yellow, ppt, npt, 1, Scalar(255), 8);
	Mat bg_white(Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), CV_8UC1, Scalar(0));
	fillPoly(bg_white, qqt, nqt, 1, Scalar(255), 8);
	Mat bg_road(Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), CV_8UC1, Scalar(0));
	fillPoly(bg_road, rrt, nrt, 1, Scalar(255), 8);

	vector<Mat> channels(3);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

	while (1) {
		int t = cap.get(CAP_PROP_POS_MSEC);
		if (t > 20000) {
			break;
		}
		if (cap.grab() == 0) break;
		cap.retrieve(frame);
		mask_white = frame.clone();
		mask_yellow = frame.clone();

		// white
		GaussianBlur(mask_white, mask_white, Size(3, 3), 0, 0);
		cvtColor(mask_white, mask_white, CV_BGR2GRAY);
		adaptiveThreshold(mask_white, mask_white, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, -15);
		dilate(mask_white, mask_white, element);
		white = Scalar::all(0);
		mask_white.copyTo(white, bg_white);
		foreground1 = Scalar::all(0);
		frame.copyTo(foreground1, white);

		// yellow	
		cvtColor(mask_yellow, mask_yellow, CV_BGR2YCrCb);
		GaussianBlur(mask_yellow, mask_yellow, Size(3, 3), 0, 0);
		split(mask_yellow, channels);

		adaptiveThreshold(channels[2], channels[2], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 35, 3); 
		yellow = Scalar::all(0);
		channels[2].copyTo(yellow, bg_yellow);
		foreground2 = Scalar::all(0);
		frame.copyTo(foreground2, yellow);
		// white + yellow
		foreground = foreground1 + foreground2;
		foreground.copyTo(result, bg_road);

		imshow("result", result);

		waitKey(33);
	}

	waitKey(0);
	return 0;
}


