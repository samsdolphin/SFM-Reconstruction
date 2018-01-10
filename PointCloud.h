#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include <stdio.h>
#include "Triangulation.h"

struct Colours
{
	int red, green, blue;
};

struct Relationship
{
	cv::Point2d *P2;
	cv::Point3d *P3;
};

typedef Relationship Entry;

void fillPoints(std::vector<cv::DMatch> goodMatches, std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2, std::vector<cv::Point2f> &points1, 
	std::vector<cv::Point2f> &points2);

void getPointColor(std::vector<cv::Point2f> points1, cv::Mat BaseImageLeft, std::vector<Colours> &colours);

Matx34d tableProcess(std::vector<Entry> &table, int &entry_num, Matx34d P1, std::vector<cv::Point2f> newKeyPoints, std::vector<cv::Point2f> oldKeyPoints, cv::Mat K);

void writePointCloud(std::vector<SpacePoint> pointCloud, std::vector<Colours> colours);

void registerPointCloud(std::vector<SpacePoint> pointCloud, std::vector<cv::Point2f> points2, int &entry_num, std::vector<Entry> &table);

#endif