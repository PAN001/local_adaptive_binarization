/**************************************************************
 * Binarization with several methods
 * (0) Niblacks method
 * (1) Sauvola & Co.
 *     ICDAR 1997, pp 147-152
 * (2) by myself - Christian Wolf
 *     Research notebook 19.4.2001, page 129
 * (3) by myself - Christian Wolf
 *     20.4.2007
 *
 * See also:
 * Research notebook 24.4.2001, page 132 (Calculation of s)
 **************************************************************/

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

enum NiblackVersion
{
	NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

#define BINARIZEWOLF_VERSION	"2.4 (August 1st, 2014)"
#define BINARIZEWOLF_DEFAULTDR	128

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)



// *************************************************************
double calcLocalStats (Mat &im, Mat &im_sum, Mat &im_sum_sq, Mat &map_m, Mat &map_s, int winx, int winy) {    
    timespec startTime;
    getTimeMonotonic(&startTime);

    // Mat im_sum, im_sum_sq;
    // cv::integral(im,im_sum,im_sum_sq,CV_64F); // TODO: no need to calculated this everytime

    // timespec endTime;
    // getTimeMonotonic(&endTime);
    // cout << "cv::integral Time: " << diffclock(startTime, endTime) << "ms." << endl;

    // double m,s,max_s,sum,sum_sq;  
    int wxh   = winx/2;
    int wyh   = winy/2;
    int x_firstth= wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    double winarea = winx*winy;

    double max_s = 0;

    // cout << "  --outerloop size: " << y_lastth - y_firstth << endl;
    // cout << "  --innerloop size: " << im.cols-winx - 1 << endl;
    for(int j = y_firstth ; j<=y_lastth; j++) { 
        double m,s,sum,sum_sq;  
        sum = sum_sq = 0;

        // sum of the window
        sum = im_sum.at<double>(j-wyh+winy,winx) - im_sum.at<double>(j-wyh,winx) - im_sum.at<double>(j-wyh+winy,0) + im_sum.at<double>(j-wyh,0);
        sum_sq = im_sum_sq.at<double>(j-wyh+winy,winx) - im_sum_sq.at<double>(j-wyh,winx) - im_sum_sq.at<double>(j-wyh+winy,0) + im_sum_sq.at<double>(j-wyh,0);

        m  = sum / winarea;
        s  = sqrt ((sum_sq - m*sum)/winarea);
        if (s > max_s) max_s = s;

        map_m.fset(x_firstth, j, m);
        map_s.fset(x_firstth, j, s);

        // Shift the window, add and remove   new/old values to the histogram
        for(int i=1 ; i <= im.cols-winx; i++) {
            // Remove the left old column and add the right new column
            sum -= im_sum.at<double>(j-wyh+winy,i) - im_sum.at<double>(j-wyh,i) - im_sum.at<double>(j-wyh+winy,i-1) + im_sum.at<double>(j-wyh,i-1);
            sum += im_sum.at<double>(j-wyh+winy,i+winx) - im_sum.at<double>(j-wyh,i+winx) - im_sum.at<double>(j-wyh+winy,i+winx-1) + im_sum.at<double>(j-wyh,i+winx-1);

            sum_sq -= im_sum_sq.at<double>(j-wyh+winy,i) - im_sum_sq.at<double>(j-wyh,i) - im_sum_sq.at<double>(j-wyh+winy,i-1) + im_sum_sq.at<double>(j-wyh,i-1);
            sum_sq += im_sum_sq.at<double>(j-wyh+winy,i+winx) - im_sum_sq.at<double>(j-wyh,i+winx) - im_sum_sq.at<double>(j-wyh+winy,i+winx-1) + im_sum_sq.at<double>(j-wyh,i+winx-1);

            m  = sum / winarea;
            s  = sqrt ((sum_sq - m*sum)/winarea);
            if (s > max_s) max_s = s;

            map_m.fset(i+wxh, j, m);
            map_s.fset(i+wxh, j, s);
        }
    }


    timespec endTime;
    getTimeMonotonic(&endTime);
    cout << "  --calcLocalStats Time: " << diffclock(startTime, endTime) << "ms." << endl;

    return max_s;
}

/**********************************************************
* The binarization routine
**********************************************************/
// https://pdfs.semanticscholar.org/0695/bb92a3301be9343433334692bd54c31a8233.pdf:
// Niblack’s algorithm calculates a threshold sur- face by gliding a rectangular window across the image. 
// The threshold T for the center pixel of the window is computed using the mean m and the variance s of the gray values in the window:
// T = m + k · s, where k is a constant set to −0.2.

__device__ __inline__ void set_color(unsigned char* input, unsigned char* output, int row_idx, int col_idx, double th, int img_width) {
    int idx = row_idx * img_width + col_idx;
    if(input[idx] >= th) {
        output[idx] = 255; // set to white
    }
    else {
        output[idx] = 0;
    }
}

__global__ void NiblackSauvolaWolfJolionCuda(unsigned char* input, float* im_sum, float* im_sum_sq, double min_I, double max_I, unsigned char* output,
    int winx, int winy, double k, double max_s, int img_width, int img_height) {
    double th=0;
    // double min_I, max_I;
    int wxh = winx/2;
    int wyh = winy/2;
    int x_firstth= wxh;
    int x_lastth = img_width-wxh-1;
    int y_lastth = img_height-wyh-1;
    int y_firstth= wyh;
    int mx, my;

    // Mat thsurf (im.rows, im.cols, CV_32F);
            
    // Create the threshold surface, including border processing
    // ----------------------------------------------------

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x; // row index
    row_idx += y_firstth;

    if(row_idx > y_lastth)
        return;

    // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
    for (int i=0 ; i <= img_width-winx; i++) {
        m = map_m[row_idx * img_width + i + wxh]
        s = map_s[row_idx * img_width + i + wxh]
        
        th = m + k * (s/max_s-1) * (m-min_I);
        set_color(input, output, row_idx, i+wxh, th, img_width)

        if (i==0) {
            // LEFT BORDER
            for (int i=0; i<=x_firstth; ++i)
                set_color(input, output, row_idx, i, th, img_width)

            // LEFT-UPPER CORNER
            if (j==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    for (int i=0; i<=x_firstth; ++i)
                        set_color(input, output, u, i, th, img_width)

            // LEFT-LOWER CORNER
            if (j==y_lastth)
                for (int u=y_lastth+1; u<img_height; ++u)
                    for (int i=0; i<=x_firstth; ++i)
                        set_color(input, output, u, i, th, img_width)
        }

        // UPPER BORDER
        if (j==y_firstth)
            for (int u=0; u<y_firstth; ++u)
                set_color(input, output, u, i+wxh, th, img_width)

        // LOWER BORDER
        if (j==y_lastth)
            for (int u=y_lastth+1; u<img_height; ++u)
                set_color(input, output, row_idx, i+wxh, th, img_width)
    }

    // RIGHT BORDER
    for (int i=x_lastth; i<img_width; ++i)
        set_color(input, output, row_idx, i, th, img_width)

    // RIGHT-UPPER CORNER
    if (row_idx==y_firstth)
        for (int u=0; u<y_firstth; ++u)
            for (int i=x_lastth; i<img_width; ++i)
                set_color(input, output, u, i, th, img_width)

    // RIGHT-LOWER CORNER
    if (row_idx==y_lastth)
        for (int u=y_lastth+1; u<img_height; ++u)
            for (int i=x_lastth; i<img_width; ++i)
                set_color(input, output, u, i, th, img_width)
}

void NiblackSauvolaWolfJolionWrapper(Mat input, Mat output, int winx, int winy, double k) {
    Mat im_sum, im_sum_sq;
    integral(input, im_sum, im_sum_sq, CV_64F);

    double min_I, max_I;
    minMaxLoc(input, &min_I, &max_I);

    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros(im.rows, im.cols, CV_32F); // mean of the gray values in the window
    Mat map_s = Mat::zeros(im.rows, im.cols, CV_32F); // variance of the gray values in the window
    max_s = calcLocalStats(im, im_sum, im_sum_sq, map_m, map_s, winx, winy);

    //Calculate total number of bytes of input and output image
    const int inputBytes = input.cols * input.rows;
    const int outputBytes = output.cols * output.rows;
    const int sumBytes = im_sum.cols * im_sum.rows;
    const int sumSqBytes = im_sum_sq.cols * im_sum_sq.rows;

    unsigned char *d_input, *d_output;
    float *d_sum, *d_sum_sq;

    //Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input,inputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output,outputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_sum,sumBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_sum_sq,sumSqBytes),"CUDA Malloc Failed");

    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_sum,im_sum.ptr(),sumBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_sum_sq,d_sum_sq.ptr(),sumSqBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    //1-d allocation
    int wyh = winy/2;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    int total_cnt = y_lastth - y_firstth + 1;
    const dim3 block(256, 1, 1);
    const dim3 grid((total_cnt + block.x - 1) / block.x, 1, 1);

    //Launch the binarization kernel
    NiblackSauvolaWolfJolionCuda(d_input, d_sum, d_sum_sq, min_I, max_I, output, winx, winy, k, max_s, input.cols, input.rows)

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(output.ptr(),d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    //Free the device memory
    SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_sum),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_sum_sq),"CUDA Free Failed");
}
