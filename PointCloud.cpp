#include <iostream>
#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
#include "PointCloud.h"

using namespace std;
using namespace cv;


void fillPoints(vector<DMatch> goodMatches, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<Point2f> &points1, 
	vector<Point2f> &points2)
{
	for (vector<DMatch>::const_iterator it = goodMatches.begin(); it != goodMatches.end(); ++it)
	{
		float x = keypoints1[it->queryIdx].pt.x; // Get the position of left keypoints
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(Point2f(x, y));

		x = keypoints2[it->trainIdx].pt.x; // Get the position of right keypoints
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(Point2f(x, y));
	}
}

void getPointColor(vector<Point2f> points1, Mat BaseImageLeft, vector<Colours> &colours)
{
	for (int i = 0; i < points1.size(); i++)
	{
		int x = int(points1.at(i).x + 0.5);
		int y = int(points1.at(i).y + 0.5);

		Point3_<uchar> *p = BaseImageLeft.ptr<Point3_<uchar> >(y, x);
		Colours pointColour;

		pointColour.blue = int(p->x);
		pointColour.green = int(p->y);
		pointColour.red = int(p->z);

		colours.push_back(pointColour);
	}
}

Matx34d tableProcess(vector<Entry> &table, int &entry_num, Matx34d P1, vector<Point2f> newKeyPoints, vector<Point2f> oldKeyPoints, Mat K)
{
	vector<Point2d> foundPoints2d;
	vector<Point3d> foundPoints3d;

	for (int i = 0; i < oldKeyPoints.size(); i++)
	{
		bool found = false;
		int index = 0;
		for (int j = 0; j < entry_num; j++)
			if (table[j].P2->x == oldKeyPoints.at(i).x && table[j].P2->y == oldKeyPoints.at(i).y)
			{
				found = true;
				index = j;
			}

		if (found)
		{
			Point3d newPoint;

			newPoint.x = table[index].P3->x;
			newPoint.y = table[index].P3->y;
			newPoint.z = table[index].P3->z;

			Point2d newPoint2;

			newPoint2.x = newKeyPoints.at(i).x;
			newPoint2.y = newKeyPoints.at(i).y;
			foundPoints3d.push_back(newPoint);
			foundPoints2d.push_back(newPoint2);

			if (entry_num >= table.size())
				table.resize(500 + table.size());

			Entry e;
			e.P2 = &newPoint2;
			e.P3 = &newPoint;
			table[entry_num] = e;
			entry_num++;
		}
	}

	int size = foundPoints3d.size();

	Mat_<double> found3dPoints(size, 3);
	Mat_<double> found2dPoints(size, 2);

	for (int i = 0; i < size; i++)
	{

		found3dPoints(i, 0) = foundPoints3d.at(i).x;
		found3dPoints(i, 1) = foundPoints3d.at(i).y;
		found3dPoints(i, 2) = foundPoints3d.at(i).z;

		found2dPoints(i, 0) = foundPoints2d.at(i).x;
		found2dPoints(i, 1) = foundPoints2d.at(i).y;

	}

	Mat P0(P1);

	Mat r(P0, Rect(0, 0, 3, 3));
	Mat t(P0, Rect(3, 0, 1, 3));

	Mat r_rog;
	cv::Rodrigues(r, r_rog);

	Mat dist = Mat::zeros(1, 4, CV_32F);
	double _dc[] = { 0, 0, 0, 0 };

	cv::solvePnP(found3dPoints, found2dPoints, K, Mat(1, 4, CV_64FC1, _dc), r_rog, t, false);

	Mat_<double> R1(3, 3);
	Mat_<double> t1(t);

	cv::Rodrigues(r_rog, R1);

	Mat camera = (Mat_<double>(3, 4) << R1(0, 0), R1(0, 1), R1(0, 2), t1(0),
										R1(1, 0), R1(1, 1), R1(1, 2), t1(1),
										R1(2, 0), R1(2, 1), R1(2, 2), t1(2));

	return Matx34d(camera);
}

void writePointCloud(vector<SpacePoint> pointCloud, vector<Colours> colours)
{
	ofstream outfile("pointcloud.ply");
	outfile << "ply\n" << "format ascii 1.0\n" << "element face 0\n";
	outfile << "property list uchar int vertex_indices\n" << "element vertex " << pointCloud.size() << "\n";
	outfile << "property float x\n" << "property float y\n" << "property float z\n";
	outfile << "property uchar diffuse_red\n" << "property uchar diffuse_green\n" << "property uchar diffuse_blue\n";
	outfile << "end_header\n" << "0 0 0 255 0 0\n";

	for (int i = 0; i < pointCloud.size(); i++)
	{
		outfile << pointCloud.at(i).point.x << " ";
		outfile << pointCloud.at(i).point.y << " ";
		outfile << pointCloud.at(i).point.z << " ";
		outfile << colours.at(i).blue << " ";
		outfile << colours.at(i).green << " ";
		outfile << colours.at(i).red << " ";
		outfile << "\n";
	}

	outfile.close();
}

void registerPointCloud(vector<SpacePoint> pointCloud, vector<Point2f> points2, int &entry_num, vector<Entry> &table)
{
	int threeD_Start = pointCloud.size() - points2.size();

	for (int i = 0; i < points2.size(); i++)
	{
		Point2d *twoD = (Point2d *)malloc(sizeof(Point2d));
		Point3d *threeD = (Point3d *)malloc(sizeof(Point3d));

		twoD->x = points2.at(i).x;
		twoD->y = points2.at(i).y;

		threeD->x = pointCloud.at(threeD_Start + i).point.x;
		threeD->y = pointCloud.at(threeD_Start + i).point.y;
		threeD->z = pointCloud.at(threeD_Start + i).point.z;

		if (entry_num >= table.size())
			table.resize(500 + table.size());

		Entry e;
		e.P2 = twoD;
		e.P3 = threeD;
		table[entry_num] = e;
		entry_num++;
	}
}