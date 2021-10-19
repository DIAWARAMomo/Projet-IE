#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;

#include <sys/time.h>

#define PI 3.14159265
#define thetaStep 4

#define VIDEO 0

void Sobel1(Mat frame, int * filterGX,int * filterGY, int size, Mat * out, int limit){

  //Convolution Time ! (adding threshold to result might improve performances significantly)
  int step = std::floor(size/2);
  float sumX,sumY;
  for(int y=1;y<frame.cols-1;y++){
    for(int x=1;x<frame.rows-1;x++){
      sumX=0;
      sumY=0;
      /*
      sumX = -frame.at<uchar>(Point(y-1,x-1))
              -2*frame.at<uchar>(Point(y,x-1))
              -frame.at<uchar>(Point(y+1,x-1))+
              frame.at<uchar>(Point(y-1,x+1))+
              2*frame.at<uchar>(Point(y,x+1))+
              frame.at<uchar>(Point(y+1,x+1));
      sumY = -frame.at<uchar>(Point(y-1,x-1))
              -2*frame.at<uchar>(Point(y-1,x))
              -frame.at<uchar>(Point(y-1,x+1))+
              frame.at<uchar>(Point(y+1,x-1))+
              2*frame.at<uchar>(Point(y+1,x))+
              frame.at<uchar>(Point(y+1,x+1));
      */
      for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
          sumX+=filterGX[i*size+j]*frame.at<uchar>(Point(y-step+i,x-step+j));
          sumY+=filterGY[i*size+j]*frame.at<uchar>(Point(y-step+i,x-step+j));
        }
      }
      out->at<uchar>(Point(y,x)) = sqrt(pow(sumX,2)+pow(sumY,2))/4;

      if(sqrt(pow(sumX,2)+pow(sumY,2))/4<limit)
        out->at<uchar>(Point(y,x)) = 0;
      //else
        //out->at<uchar>(Point(y,x)) = 0;

    }
  }
}

void SobelED(Mat frame, Mat * out,int limit){

  if(limit>255 || limit<0)
    limit=255;
  float sumX,sumY;
  for(int y=1;y<frame.cols-1;y++){
    for(int x=1;x<frame.rows-1;x++){
      sumX=0;
      sumY=0;
      sumX = frame.at<uchar>(Point(y,x+1))-frame.at<uchar>(Point(y,x-1));
      sumY = frame.at<uchar>(Point(y+1,x))-frame.at<uchar>(Point(y-1,x));

      if(sqrt(pow(sumX,2)+pow(sumY,2))>limit){
        out->at<uchar>(Point(y,x)) = 255;
      }

      else{
        out->at<uchar>(Point(y,x)) = 0;
      }
    }
  }
}

int diff_ms(timeval t1, timeval t2)
{
    return (((t1.tv_sec - t2.tv_sec) * 1000000) +
            (t1.tv_usec - t2.tv_usec))/1000;
}

// Convert RGB image to grayscale using the luminosity method
void RGBtoGrayScale(Mat rgb, Mat* grayscale){
  std::vector<Mat> channels(3);
  split(rgb, channels);
  *grayscale = (0.07*channels[0] + 0.72*channels[1] + 0.21*channels[2]);
}

void simpleHough(Mat frame, Mat* acc, Mat *f){
  int channels = frame.channels();
  int nRows = frame.rows;
  int nCols = frame.cols * channels;
  //const uchar* image = frame.ptr();
  int step = (int)frame.step;
  int stepacc = (int)acc->step;
  if (frame.isContinuous())
  {
      nCols *= nRows;
      nRows = 1;
  }

  int i,j;
  double rho;
  for( i = 0; i < frame.rows; i++ ){
    for( j = 0; j < frame.cols; j++ ){
      if(frame.data[i * step + j]!=0){
        for(int theta=0;theta<180; theta+=thetaStep){
          rho = j*cos((double)theta*PI/180)+i*sin((double)theta*PI/180);
          if(rho!=0)
            acc->at<ushort>(Point(cvRound(rho) ,(int)cvRound(theta/thetaStep)))+=1;
        }
      }
    }
  }
  cv::Point min_loc, max_loc;
  cv::Point min_loc_old, max_loc_old;
  double min, max;
  cv::minMaxLoc(*acc, &min, &max, &min_loc_old, &max_loc_old);

  Point pt1, pt2;
  double a ,b;
  double x0, y0;
  double theta;
  acc->data[max_loc_old.y * stepacc +max_loc_old.x]=0;
  for(int i=0;i<40;i++){
    cv::minMaxLoc(*acc, &min, &max, &min_loc, &max_loc);
    if(abs(max_loc_old.x-max_loc.x)>5 || abs(max_loc_old.y-max_loc.y)>5){ //might be interesting to use that ....
      theta = (double)max_loc.y*thetaStep;
      a = cos(theta*PI/180); //compute hough inverse transform from polar to cartesian
      b = sin(theta*PI/180);
      x0 = a*max_loc.x;
      y0 = b*max_loc.x;
      pt1.x = cvRound(x0 + 1000*(-b)); //compute first point belonging to the line
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b)); //compute second point
      pt2.y = cvRound(y0 - 1000*(a));
      line( *f, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
      acc->at<ushort>(Point(max_loc.x ,max_loc.y))=0;
      max_loc_old.x = max_loc.x;
      max_loc_old.y = max_loc.y;
    }
    else{
      acc->at<ushort>(Point(max_loc.x ,max_loc.y))=0;
      i--;
    }
  }
}



int main(int argc, char** argv)
{
  timeval start, end;
  printf("OK !\n");
  char name[50];
  Mat frame;
  Mat canny;

  #if VIDEO
  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened()){  // check if we succeeded
      printf("Error capture\n");
      return -1;
  }
  cap >> frame;
  #endif
  #if VIDEO==0
  //Mat frame;
  frame = imread(argv[1], IMREAD_COLOR);

  if(! frame.data )                              // Check for invalid input
  {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    #endif


  printf("setup\n");
    int x = frame.rows;
    int y = frame.cols;
    Size pt = Size(sqrt(pow(x,2)+pow(y,2))*2,ceil(180/thetaStep));
    printf("rows : %d, cols : %d \n",frame.rows, frame.cols);
    Mat acc = Mat::zeros(pt, CV_16UC1);
    /* example of creating image and save it using opencv
    Mat exmp = Mat::zeros(Size(200,200),CV_16UC1);
    exmp.at<uchar>(Point(20,5))=255;
    exmp.at<uchar>(Point(100,25))=255;

    exmp.at<uchar>(Point(20,5))=255;
    exmp.at<uchar>(Point(100,25))=255;
    imwrite( "./test5.jpg", exmp );
    */
    printf("rho : %d, theta : %d \n",acc.cols, acc.rows); //size of the accumulation matrix
    Mat grayscale = Mat::zeros(Size(frame.rows,frame.cols),CV_8UC1);
    Mat sobel = Mat::zeros(Size(frame.cols,frame.rows),CV_8UC1);
    int filterGX[9] = {-1,0,1,-2,0,2,-1,0,1};
    int filterGY[9] = {-1,-2,-1,0,0,0,1,2,1};
    int convSize = 3;
    namedWindow("Color Frame",1);
    namedWindow("Sobel Frame",1);

    #if VIDEO==0 //if image mode selected, must have image path as first argument
    RGBtoGrayScale(frame,&grayscale);
    GaussianBlur(grayscale, grayscale, Size(9,9), 2, 2);
    Sobel1(grayscale,filterGX,filterGY,convSize, &sobel,25);
    //Canny( grayscale, sobel, 60, 60*3,3);
    //SobelED(grayscale,&sobel,20);
    simpleHough(sobel,&acc,&frame);

    imshow( "Color Frame", frame );  // Show our image with hough line detection
    imshow( "Sobel Frame", sobel );
    imshow("Acc Frame", acc);
    waitKey(0); //wait for key pressed in order to propely close all opencv windows
    #else //if video selected, no needs of arguments in the program call
    for(;;)
    {
        Mat res;
        gettimeofday(&start,NULL);
        cap >> frame; // get a new frame from camera
        acc = Mat::zeros(pt, CV_16UC1); //16 bits for accumulatio matrix
        RGBtoGrayScale(frame,&grayscale);
        GaussianBlur(grayscale, grayscale, Size(9,9), 1.5, 1.5);
        //Sobel1(grayscale,filterGX,filterGY,3, &sobel); //sobel filter, not optimized though
        SobelED(grayscale,&sobel,20); //simple filter use for testing only, not mean to be used with hough transform
        //Canny( grayscale, sobel, 60, 60*3,3); //opencv canny filter, use to compare performances
        simpleHough(sobel,&acc,&frame);
        gettimeofday(&end,NULL);
        int ms = diff_ms(end,start);
        //normalize(acc,acc,0,255,NORM_MINMAX, CV_16UC1); //normalize mat, use at your discretion
        sprintf (name, "fps : %f", 1/((double)ms/1000));
        putText(frame, name, cv::Point(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(20,20,25), 1, LINE_AA);

        imshow( "Color Frame", frame );  // Show image with Hough line detection
        imshow( "Sobel Frame", sobel );  // show filter image
        imshow("Acc Frame", acc);
        if(waitKey(33) == 27) break;
    }
    #endif
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
