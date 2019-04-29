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
    int c;
    char *inputname, *outputname;
    outputname = "output.jpg";

    int N = 6;
    char* file_names[] = {"250_250.jpg", "400_400.jpg", "500_500.jpg", "640_640.jpg", "800_800.jpg", "1024_1024.jpg"}
    for(int i = 0;i < N;i++) {
        inputname = file_names[i];
        cout << "=========== " << inputname << endl;

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
        // // WolfJolion
        // int k = 0, win=18;
        // double WolfJolion_k = 0.05 + (k * 0.35);

        // Sauvola
        int k = 1, win = 12;
        double sauvola_k = 0.18 * k;
        NiblackSauvolaWolfJolionWrapper(input, output, win, win, sauvola_k);

        timespec endTime;
        getTimeMonotonic(&endTime);
        cout << "=========== Total time: " << diffclock(startTime, endTime) << "ms." << endl;

        // Write the tresholded file
        // cerr << "Writing binarized image to file '" << outputname << "'.\n";
        imwrite (outputname, output);

        cout << "" << endl;
        cout << "" << endl;
    }

    return 0;
}