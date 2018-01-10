#ifndef TRIANGULATION_H_
#define TRIANGULATION_H_

#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"

struct Point3D
{
	cv::Point3d point;
	int red, green, blue;
};

typedef Point3D SpacePoint;

typedef cv::Matx< double, 3, 4 > Matx34d;

std::vector<SpacePoint> Triangulation(cv::Mat BaseImageLeft, cv::Mat BaseImageRight, cv::Mat K, cv::Mat fundamentalMatrix, Matx34d &P, Matx34d &P1, std::vector<cv::Point2f> points1, 
	std::vector<cv::Point2f> points2, std::vector<SpacePoint> &pointCloud);

std::vector<SpacePoint> Triangulation(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, cv::Mat K, Matx34d P, Matx34d P1, std::vector<SpacePoint> &pointCloud);

cv::Mat_<double> IterativeTriangulation(cv::Point3d u, Matx34d P, cv::Point3d u1, Matx34d P1);

cv::Mat_<double> LinearLSTriangulation(cv::Point3d u, Matx34d P, cv::Point3d u1, Matx34d P1);

#endif