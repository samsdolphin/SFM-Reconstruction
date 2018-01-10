#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "Triangulation.h"
#include "PointCloud.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &goodMatches);
void symmetryTest(const vector<DMatch> &matches1, const vector<DMatch> &matches2, vector<DMatch>& symMatches);

vector<Entry> table;
int entry_num = 0;


int main(int argc, char** argv)
{
	string prefix = "00";
	string extension = ".png";
	string imageName1, imageName2;
	int pictureNumber1 = 3;
	int pictureNumber2 = 4;
	string stringpicturenumber1 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber1))->str();
	string stringpicturenumber2 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber2))->str();
	imageName1 = prefix + stringpicturenumber1 + extension;
	imageName2 = prefix + stringpicturenumber2 + extension;

	bool firstTwoImages = true;
	Mat K;
	Matx34d P, P1;
	vector<SpacePoint> pointCloud;
	vector<Colours> colours;

	while (pictureNumber1 < 8)
	{
		Mat BaseImageLeft = imread(imageName1, -1);
		Mat BaseImageRight = imread(imageName2, -1);

		if(!BaseImageLeft.data || !BaseImageRight.data)
			cout << "ERROR! NO IMAGE LOADED!" << endl;

		Ptr<SURF> detector = SURF::create(10);
		vector<KeyPoint> keypoints_1, keypoints_2;
		detector->detect(BaseImageLeft, keypoints_1);
		detector->detect(BaseImageRight, keypoints_2);

		Ptr<SURF> extractor = SURF::create();
		Mat descriptors_1, descriptors_2;
		extractor->compute(BaseImageLeft, keypoints_1, descriptors_1);
		extractor->compute(BaseImageRight, keypoints_2, descriptors_2);

		FlannBasedMatcher matcher;
		vector<vector<DMatch> > matches1, matches2;
		vector<DMatch> goodMatches1, goodMatches2, goodMatches, outMatches;
		matcher.knnMatch(descriptors_1, descriptors_2, matches1, 2); // find 2 nearest neighbours, match.size() = query.rowsize()
		matcher.knnMatch(descriptors_2, descriptors_1, matches2, 2);
		ratioTest(matches1, goodMatches1);
		ratioTest(matches2, goodMatches2);
		symmetryTest(goodMatches1, goodMatches2, goodMatches); // double check

		if (firstTwoImages)
		{
			Mat fundamentalMatrix;
			vector<Point2f> points1, points2;

			if (goodMatches.size() < 30)
				cerr << "ERROR: NOT ENOUGH MATCHES" << endl;
			else
			{
				fillPoints(goodMatches, keypoints_1, keypoints_2, points1, points2);
				
				vector<uchar> inliers(points1.size(), 0);
				fundamentalMatrix = findFundamentalMat(Mat(points1), Mat(points2), inliers, CV_FM_RANSAC, 3.0, 0.99); // Compute fundamental matrix using RANSAC
				
				vector<DMatch>::const_iterator	itM = goodMatches.begin(); // extract the surviving (inliers) matches
				for (vector<uchar>::const_iterator itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itM)
					if (*itIn) // it is a valid match
						outMatches.push_back(*itM);
				if (outMatches.size() < 25)
					cerr << "ERROR: NOT ENOUGH MATCHES" << endl;

				points1.clear();
				points2.clear();

				fillPoints(outMatches, keypoints_1, keypoints_2, points1, points2);

				fundamentalMatrix = findFundamentalMat(Mat(points1), Mat(points2), CV_FM_8POINT);
			}

			getPointColor(points1, BaseImageLeft, colours);

			K = (Mat_<double>(3, 3) << 2569.1, 0, 2180.3, 0, 2429.6, 1416.3, 0, 0, 1); // Fountain
			//K = (Mat_<double>(3, 3) << 4269.4, 0, 2629.6, 0, 4269.5, 1728.6, 0, 0, 1); // UAV LSK

			pointCloud = Triangulation(BaseImageLeft, BaseImageRight, K, fundamentalMatrix, P, P1, points1, points2, pointCloud);

			registerPointCloud(pointCloud, points2, entry_num, table);

			firstTwoImages = false;

			cout << "Complete processing " << imageName1 << " and " << imageName2 << endl;
		}
		else
		{
			Mat fundamentalMatrix;
			vector<Point2f> points1, points2;

			if (goodMatches.size() < 30)
				cerr << "ERROR: NOT ENOUGH MATCHES" << endl;
			else
			{
				fillPoints(goodMatches, keypoints_1, keypoints_2, points1, points2);

				vector<uchar> inliers(points1.size(), 0);
				fundamentalMatrix = findFundamentalMat(Mat(points1), Mat(points2), inliers, CV_FM_RANSAC, 3.0, 0.99);

				vector<DMatch>::const_iterator	itM = goodMatches.begin();
				for (vector<uchar>::const_iterator itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itM)
					if (*itIn)
						outMatches.push_back(*itM);
				if (outMatches.size() < 25)
					cerr << "ERROR: NOT ENOUGH MATCHES" << endl;

				points1.clear();
				points2.clear();

				fillPoints(outMatches, keypoints_1, keypoints_2, points1, points2);
			}

			P = P1;
			P1 = tableProcess(table, entry_num, P1, points2, points1, K);

			pointCloud = Triangulation(points1, points2, K, P, P1, pointCloud);

			getPointColor(points1, BaseImageLeft, colours);

			registerPointCloud(pointCloud, points2, entry_num, table);

			cout << "Complete processing " << imageName1 << " and " << imageName2 << endl;
		}

		pictureNumber1++;
		pictureNumber2++;
		stringpicturenumber1 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber1))->str();
		stringpicturenumber2 = static_cast<ostringstream*>(&(ostringstream() << pictureNumber2))->str();
		imageName1 = prefix + stringpicturenumber1 + extension;
		imageName2 = prefix + stringpicturenumber2 + extension;
	}

	cout<<"COMPLETED! Point Cloud Size: "<<pointCloud.size()<<endl;

	writePointCloud(pointCloud, colours);

	return 0;
}

void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &goodMatches)
{
	for (vector<vector<DMatch> >::iterator matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
		if (matchIterator->size() > 1)
			if ((*matchIterator)[0].distance < (*matchIterator)[1].distance * 0.8) // check distance ratio
				goodMatches.push_back((*matchIterator)[0]);
}

void symmetryTest(const vector<DMatch> &matches1, const vector<DMatch> &matches2, vector<DMatch>& symMatches)
{
	symMatches.clear();
	for (vector<DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
		for (vector<DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
			if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator2).queryIdx == (*matchIterator1).trainIdx)
				symMatches.push_back(DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
}