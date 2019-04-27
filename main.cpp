#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
// #include <cv>
// #include <highgui>
#include <opencv2/opencv.hpp>
#include "timing.h"

using namespace std;
using namespace cv;

void NiblackSauvolaWolfJolionWrapper(Mat input, Mat output, int winx, int winy, double k);


int main (int argc, char **argv)
{
    char version;
    int c;
    int winx=0, winy=0;
    float optK=0.5;
    char *inputname, *outputname, *versionstring;
    // inputname = "plate.jpg";
    inputname = "1080p.jpeg"
    outputname = "output.jpg";

    cerr << "Adaptive binarization\n"
         << "Threshold calculation: ";

    // Load the image in grayscale mode
    Mat input = imread(inputname,CV_LOAD_IMAGE_GRAYSCALE);

    if ((input.rows<=0) || (input.cols<=0)) {
        cerr << "*** ERROR: Couldn't read input image " << inputname << endl;
        exit(1);
    }

    timespec startTime;
    getTimeMonotonic(&startTime);
    // Mat im_sum, im_sum_sq;
    // integral(input, im_sum, im_sum_sq, CV_64F);

    // timespec integralEndTime;
    // getTimeMonotonic(&integralEndTime);
    // cout << "  --cv::integral Time: " << diffclock(startTime, integralEndTime) << "ms." << endl;

    // timespec minMaxLocStartTime;
    // getTimeMonotonic(&minMaxLocStartTime);
    // double min_I, max_I;
    // minMaxLoc(input, &min_I, &max_I);

    // timespec minMaxLocEndTime;
    // getTimeMonotonic(&minMaxLocEndTime);
    // cout << "  --cv::minMaxLoc Time: " << diffclock(minMaxLocStartTime, minMaxLocEndTime) << "ms." << endl;

    // Threshold
    Mat output (input.rows, input.cols, CV_8U);
    // NiblackSauvolaWolfJolion (input, output, versionCode, winx, winy, optK, 128);
    int k = 0, win=18;
    NiblackSauvolaWolfJolionWrapper(input, output, win, win, 0.05 + (k * 0.35));

    timespec endTime;
    getTimeMonotonic(&endTime);
    cout << "=========== Time: " << diffclock(startTime, endTime) << "ms." << endl;

    // Write the tresholded file
    cerr << "Writing binarized image to file '" << outputname << "'.\n";
    imwrite (outputname, output);

    return 0;
}