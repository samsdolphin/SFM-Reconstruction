#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "Triangulation.h"

using namespace std;
using namespace cv;


vector<SpacePoint> Triangulation(Mat BaseImageLeft, Mat BaseImageRight, Mat K, Mat fundamentalMatrix, Matx34d &P, Matx34d &P1, vector<Point2f> points1, 
	vector<Point2f> points2, vector<SpacePoint> &pointCloud)
{
	double pX = BaseImageLeft.cols / 2.0;
	double pY = BaseImageRight.rows / 2.0;

	Mat_<double> E = K.t() * fundamentalMatrix * K; // E = (K').transpose() * F * K

	SVD svd(E, SVD::MODIFY_A);
	Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1); // equation $9.13 page 258
	Matx33d Wt(0, 1, 0, -1, 0, 0, 0, 0, 1);

	Mat_<double> R1 = svd.u * Mat(W) * svd.vt; // equation $9.14 page 258
	Mat_<double> R2 = svd.u * Mat(Wt) * svd.vt;
	Mat_<double> t1 = svd.u.col(2); // t = U(0, 0, 1).transpose() = u3 page 259
	Mat_<double> t2 = -svd.u.col(2);

	Mat Ptemp = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	P = Matx34d(Ptemp);
	// TODO: need to make sure det(P1si) > 0
	Mat P1s1 = (Mat_<double>(3, 4) << R1(0, 0), R1(0, 1), R1(0, 2), t2(0), R1(1, 0), R1(1, 1), R1(1, 2), t2(1), R1(2, 0), R1(2, 1), R1(2, 2), t2(2));
	P1 = Matx34d(P1s1);

	pointCloud = Triangulation(points1, points2, K, P, P1, pointCloud);

	return pointCloud;
}

vector<SpacePoint> Triangulation(vector<Point2f> points1, vector<Point2f> points2, Mat K, Matx34d P, Matx34d P1, vector<SpacePoint> &pointCloud)
{
	// http://www.ics.uci.edu/~dramanan/teaching/cs217_spring09/lec/stereo.pdf
	Mat kInverse = K.inv();

	vector<SpacePoint> tempCloud = pointCloud;
	for (int i = 0; i < points1.size(); i++)
	{
		Point3d point3D1(points1.at(i).x, points1.at(i).y, 1);
		Mat_<double> mapping3D1 = kInverse * Mat_<double>(point3D1); // K.inverse() * (x, y, 1).transpose() = (X, Y, Z).transpose()
		point3D1.x = mapping3D1(0);
		point3D1.y = mapping3D1(1);
		point3D1.z = mapping3D1(2);

		Point3d point3D2(points2.at(i).x, points2.at(i).y, 1);
		Mat_<double> mapping3D2 = kInverse * Mat_<double>(point3D2);
		point3D2.x = mapping3D2(0);
		point3D2.y = mapping3D2(1);
		point3D2.z = mapping3D2(2);

		Mat_<double> X = IterativeTriangulation(point3D1, P, point3D2, P1);

		SpacePoint Location3D;
		Location3D.point.x = X(0);
		Location3D.point.y = X(1);
		Location3D.point.z = X(2);

		tempCloud.push_back(Location3D);
	}

	pointCloud = tempCloud;

	return tempCloud;
}

Mat_<double> IterativeTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
	double wi = 1, wi1 = 1;
	Mat_<double> X(4, 1);

	Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
	X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

	for (int i = 0; i<10; i++) 
	{ //Hartley suggests 10 iterations at most		

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if (fabsf(wi - p2x) <= 0.0001 && fabsf(wi1 - p2x1) <= 0.0001) 
			break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A(	(u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
					(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
					(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
					(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1);

		Mat_<double> B =(Mat_<double>(4, 1) <<  -(u.x*P(2, 3) - P(0, 3)) / wi,
												-(u.y*P(2, 3) - P(1, 3)) / wi,
												-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
												-(u1.y*P1(2, 3) - P1(1, 3)) / wi1);

		solve(A, B, X_, DECOMP_SVD);

		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}

	return X;
}

Mat_<double> LinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1)
{
	//	http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf
	//	solve || D*X || = 0

	Matx43d A;

	A <<u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2);

	Matx41d B;

	B <<-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3));

	Mat_<double> X;
	
	solve(A, B, X, DECOMP_SVD);

	return X;
}