#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "core_msgs/ball_position.h"
#include "opencv2/opencv.hpp"
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <math.h>

using namespace cv;
using namespace std;

void on_low_y_thresh_trackbar_red(int, void *);
void on_high_y_thresh_trackbar_red(int, void *);
void on_low_cr_thresh_trackbar_red(int, void *);
void on_high_cr_thresh_trackbar_red(int, void *);
void on_low_cb_thresh_trackbar_red(int, void *);
void on_high_cb_thresh_trackbar_red(int, void *);

//빨간색 tarck bar의 HSV range에서 초기 값을 지정
int low_y_r=0, low_cr_r=0, low_cb_r=128;
int high_y_r=255, high_cr_r=45, high_cb_r=255;

void on_low_h_thresh_trackbar_blue(int, void *);
void on_high_h_thresh_trackbar_blue(int, void *);
void on_low_s_thresh_trackbar_blue(int, void *);
void on_high_s_thresh_trackbar_blue(int, void *);
void on_low_v_thresh_trackbar_blue(int, void *);
void on_high_v_thresh_trackbar_blue(int, void *);

//파란색 track bar의 HSV range의 초기 값을 지정
int low_h_b=96, low_s_b=183, low_v_b=61;
int high_h_b=120, high_s_b=255, high_v_b=255;

void on_low_h_thresh_trackbar_green(int, void *);
void on_high_h_thresh_trackbar_green(int, void *);
void on_low_s_thresh_trackbar_green(int, void *);
void on_high_s_thresh_trackbar_green(int, void *);
void on_low_v_thresh_trackbar_green(int, void *);
void on_high_v_thresh_trackbar_green(int, void *);

int low_h_g=33, low_s_g=65, low_v_g=109;
int high_h_g=88, high_s_g=222, high_v_g=255;
//string 문자열로 바꾸기 위한 함수를 불러옴
string intToString(int n);
string floatToString(float f);

//erode&dilate를 위한 morphOps 함수 불러옴
void morphOps(Mat &thresh);

//pixel 값을 position 값으로 바꾸는 함수 불러옴
vector<float> pixel2point(Point center, int radius);

//red image processing을 위한 함수(canny edge method) 불러옴
void on_canny_edge_trackbar_red(int, void *);
void on_canny_edge_trackbar_blue(int, void *);
void on_canny_edge_trackbar_green(int, void *);

//canny edge를 실행시키기 위해 필요한 변수 값 지정
int lowThreshold_r = 100;
int ratio_r = 3;
int kernel_size_r = 3;

int lowThreshold_g = 100;
int ratio_g = 3;
int kernel_size_g = 3;

int lowThreshold_b = 100;
int ratio_b = 3;
int kernel_size_b = 3;

//ball의 최소 detecting 가능 값을 정의
float fball_radius = 0.0375 ; // meter

//camera calibration을 해주기 위해 웹캠의 특성 값을 넣어줌
Mat distCoeffs;
float intrinsic_data[9] = {651.640791,  0, 373.068155, 0, 644.794361, 220.023746, 0, 0, 1};
float distortion_data[5] = {0.022322, -0.100660, -0.007053, 0.021342, 0};

//글자 크기 설정
double fontScale = 2;
int thickness = 3;
String text ;

//최소 tracking ball size를 지정
int iMin_tracking_ball_size = 3;
Point2f center_r1[3];
float radius_r1[3];
Point2f center_r2[3];
float radius_r2[3];
int loop=0;

Mat buffer(320,240,CV_8UC1);
ros::Publisher red_pub;
ros::Publisher blue_pub;
ros::Publisher green_pub;
ros::Publisher pub_markers;

void ball_detect(){
	loop++;
  //각각의 변수를 선언한다
	Mat frame, hsv_frame,
	hsv_frame_blue, hsv_frame_green,
	hsv_frame_red_blur, hsv_frame_blue_blur, hsv_frame_green_blur,
	hsv_frame_red_canny, hsv_frame_blue_canny, hsv_frame_green_canny, result;

	//calibration parameter를 위한 matrix 생성
	Mat calibrated_frame;
	Mat intrinsic = Mat(3,3, CV_32FC1);
	Mat distCoeffs;
	//위에서 지정한 값을 calibration 값으로 지정
	intrinsic = Mat(3, 3, CV_32F, intrinsic_data);
	distCoeffs = Mat(1, 5, CV_32F, distortion_data);

	vector<Vec4i> hierarchy_r;
	vector<Vec4i> hierarchy_b;
	vector<Vec4i> hierarchy_g;

	//빨간색, 파란색의 테두리 값을 선언한다
	vector<vector<Point> > contours_r;
	vector<vector<Point> > contours_b;
	vector<vector<Point> > contours_g;

	//원사진, detecting된 사진, 빨간색 파란색에 대한 track bar, canny edge를 실행한 결과가 나오는 window를 생성
    // namedWindow("Video Capture", WINDOW_NORMAL);
    // namedWindow("Object Detection_YCbCr_Red", WINDOW_NORMAL);
    // namedWindow("Object Detection_HSV_Blue", WINDOW_NORMAL);
    // namedWindow("Object Detection_HSV_Green", WINDOW_NORMAL);
    // namedWindow("Canny Edge for Red Ball", WINDOW_NORMAL);
    // namedWindow("Canny Edge for Blue Ball", WINDOW_NORMAL);
    // namedWindow("Canny Edge for Green Ball", WINDOW_NORMAL);
    // namedWindow("Result", WINDOW_NORMAL);
		//
    //원 사진, detecting된 사진, 빨간색 파란색에 대한 trackbar, canny edge method를 실행한 결과가 나오는 window 각각의 위치에 생성함
    // moveWindow("Video Capture",             50,  0);
    // moveWindow("Object Detection_YCbCr_Red",  50,370);
    // moveWindow("Object Detection_HSV_Blue",470,370);
    // moveWindow("Object Detection_HSV_Green",890,370);
    // moveWindow("Canny Edge for Red Ball",   50,730);
    // moveWindow("Canny Edge for Blue Ball", 470,730);
    // moveWindow("Canny Edge for Green Ball", 890,730);
    // moveWindow("Result", 470, 0);
		//
    //빨간색 detecting을 위한 track bar 생성
    // createTrackbar("Low Y","Object Detection_YCbCr_Red", &low_y_r, 255,
    // on_low_y_thresh_trackbar_red);
    // createTrackbar("High Y","Object Detection_YCbCr_Red", &high_y_r, 255,
    // on_high_y_thresh_trackbar_red);
    // createTrackbar("Low Cr","Object Detection_YCbCr_Red", &low_cr_r, 255,
    // on_low_cr_thresh_trackbar_red);
    // createTrackbar("High Cr","Object Detection_YCbCr_Red", &high_cr_r, 255,
    // on_high_cr_thresh_trackbar_red);
    // createTrackbar("Low Cb","Object Detection_YCbCr_Red", &low_cb_r, 255,
    // on_low_cb_thresh_trackbar_red);
    // createTrackbar("High Cb","Object Detection_YCbCr_Red", &high_cb_r, 255,
    // on_high_cb_thresh_trackbar_red);
		//
    // //파란색 detecting을 위한 track bar 생성
    // createTrackbar("Low H","Object Detection_HSV_Blue", &low_h_b, 180,
    // on_low_h_thresh_trackbar_blue);
    // createTrackbar("High H","Object Detection_HSV_Blue", &high_h_b, 180,
    // on_high_h_thresh_trackbar_blue);
    // createTrackbar("Low S","Object Detection_HSV_Blue", &low_s_b, 255,
    // on_low_s_thresh_trackbar_blue);
    // createTrackbar("High S","Object Detection_HSV_Blue", &high_s_b, 255,
    // on_high_s_thresh_trackbar_blue);
    // createTrackbar("Low V","Object Detection_HSV_Blue", &low_v_b, 255,
    // on_low_v_thresh_trackbar_blue);
    // createTrackbar("High V","Object Detection_HSV_Blue", &high_v_b, 255,
    // on_high_v_thresh_trackbar_blue);
		//
    // //초록색 detecting을 위한 track bar 생성
    // createTrackbar("Low H","Object Detection_HSV_Green", &low_h_g, 180,
    // on_low_h_thresh_trackbar_green);
    // createTrackbar("High H","Object Detection_HSV_Green", &high_h_g, 180,
    // on_high_h_thresh_trackbar_green);
    // createTrackbar("Low S","Object Detection_HSV_Green", &low_s_g, 255,
    // on_low_s_thresh_trackbar_green);
    // createTrackbar("High S","Object Detection_HSV_Green", &high_s_g, 255,
    // on_high_s_thresh_trackbar_green);
    // createTrackbar("Low V","Object Detection_HSV_Green", &low_v_g, 255,
    // on_low_v_thresh_trackbar_green);
    // createTrackbar("High V","Object Detection_HSV_Green", &high_v_g, 255,
    // on_high_v_thresh_trackbar_green);


      if(buffer.size().width==320){ //if the size of the image is 320x240, then resized it to 640x480
        cv::resize(buffer, frame, cv::Size(640, 480));
    }
    else{
        frame = buffer;
    }

      // ideal coordinates로 변환, frame 이미지를 웹캠의 설정 값으로 calibrated 해준 이미지를 calibrated_frame으로 저장합니다.
        undistort(frame, calibrated_frame, intrinsic, distCoeffs);

		// 새로운 Mat m을 선언, 메모리를 추가로 할당
        result = calibrated_frame.clone();

		// medianBlur 라는 이미지 프로세싱 적용
        medianBlur(calibrated_frame, calibrated_frame, 3);

	Mat  maskYCB_r, resultYCB, brightYCB, resultYCB_r;
        brightYCB = calibrated_frame.clone();
        Scalar maxYCB_r = Scalar(high_y_r,high_cr_r,high_cb_r);
        Scalar minYCB_r = Scalar(low_y_r,low_cr_r,low_cb_r);
        inRange(brightYCB, minYCB_r, maxYCB_r, maskYCB_r);
        bitwise_and(brightYCB, brightYCB, resultYCB, maskYCB_r);
        resultYCB_r = resultYCB.clone();

		// BGR 이미지를 HSV 이미지로 변환하여 hsv_frame 으로 저장
        cvtColor(calibrated_frame, hsv_frame, cv::COLOR_BGR2HSV);

        // 이미지를 두 스칼라 값을 경계로 가지는 이미지로 변환하여 지정한다
        inRange(hsv_frame,Scalar(low_h_b,low_s_b,low_v_b),Scalar(high_h_b,high_s_b,high_v_b),hsv_frame_blue);
        inRange(hsv_frame,Scalar(low_h_g,low_s_g,low_v_g),Scalar(high_h_g,high_s_g,high_v_g),hsv_frame_green);


		// 각 색깔의 지정한 범위 내의 이미지만 인식하도록 명령
        morphOps(resultYCB_r);
        morphOps(hsv_frame_blue);
        morphOps(hsv_frame_green);

		// image filtering 적용
        GaussianBlur(resultYCB_r, hsv_frame_red_blur, cv::Size(9, 9), 2, 2);
        GaussianBlur(hsv_frame_blue, hsv_frame_blue_blur, cv::Size(9, 9), 2, 2);
        GaussianBlur(hsv_frame_green, hsv_frame_green_blur, cv::Size(9, 9), 2, 2);

		// 지정한 경계로 canny image processing 진행
        Canny(hsv_frame_red_blur, hsv_frame_red_canny, lowThreshold_r, lowThreshold_r*ratio_r, kernel_size_r);
        Canny(hsv_frame_blue_blur, hsv_frame_blue_canny, lowThreshold_b, lowThreshold_b*ratio_b, kernel_size_b);
        Canny(hsv_frame_green_blur, hsv_frame_green_canny, lowThreshold_g, lowThreshold_g*ratio_g, kernel_size_g);

		// findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point())에 맞게 적용하여 경계 도출
	findContours(hsv_frame_red_canny, contours_r, hierarchy_r, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
        findContours(hsv_frame_blue_canny, contours_b, hierarchy_b, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
	findContours(hsv_frame_green_canny, contours_g, hierarchy_g, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));


		// 경계 크기를 정의
        vector<vector<Point> > contours_r_poly( contours_r.size() );
        vector<vector<Point> > contours_b_poly( contours_b.size() );
        vector<vector<Point> > contours_g_poly( contours_g.size() );
        // 경계의 중심을 정의
        vector<Point2f>center_r( contours_r.size() );
        vector<Point2f>center_b( contours_b.size() );
        vector<Point2f>center_g( contours_g.size() );
        // 경계의 반지름을 정의
        vector<float>radius_r( contours_r.size() );
        vector<float>radius_b( contours_b.size() );
        vector<float>radius_g( contours_g.size() );

	core_msgs::ball_position r_msg;  //create a message for ball positions
	core_msgs::ball_position b_msg;
	core_msgs::ball_position g_msg;
	int z=0;
		// 빨간색의 사이즈를 경계의 크기까지 점점 늘려나감
        for( size_t i = 0; i < contours_r.size(); i++ ){
            // 3의 정확도로 polygon 경계로 근사
			approxPolyDP( contours_r[i], contours_r_poly[i], 3, true );
            // 2D 점들의 최소한의 원의 영역을 찾는다
			minEnclosingCircle( contours_r_poly[i], center_r[i], radius_r[i] );
        }
		// 파란색의 사이즈를 경계의 크기까지 점점 늘려나감
        for( size_t i = 0; i < contours_b.size(); i++ ){
            // 3의 정확도로 polygon 경계로 근사
			approxPolyDP( contours_b[i], contours_b_poly[i], 3, true );
            // 2D 점들의 최소한의 원의 영역을 찾는다
			minEnclosingCircle( contours_b_poly[i], center_b[i], radius_b[i] );
        }
	for( size_t i = 0; i < contours_g.size(); i++ ){
	    // 3의 정확도로 polygon 경계로 근사
	    approxPolyDP( contours_g[i], contours_g_poly[i], 3, true );
	    // 2D 점들의 최소한의 원의 영역을 찾는다
	    minEnclosingCircle( contours_g_poly[i], center_g[i], radius_g[i] );
	}
	for( size_t i = 0; i < contours_g.size(); i++ ){
		            int j=0;
		            int w=0;
		            while ( j < contours_g.size()){
		                if (i==j){
		                    j++;
		                }
		                // 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
		                if ( sqrt((center_g[i].x-center_g[j].x)*(center_g[i].x-center_g[j].x)+(center_g[i].y-center_g[j].y)*(center_g[i].y-center_g[j].y)) < (radius_g[i] + radius_g[j]) && radius_g[i] < radius_g[j]){
		                    j++;
		                    w++;}
		                else {
		                    j++;}
		            }
		            if ( w==0 && radius_g[i] > iMin_tracking_ball_size){
				z++;
		            }
		}

      r_msg.img_x.resize(contours_r.size());
      r_msg.img_y.resize(contours_r.size());

      b_msg.img_x.resize(contours_b.size());
      b_msg.img_y.resize(contours_b.size());

      g_msg.img_x.resize(z);
      g_msg.img_y.resize(z);


        int f=-1;
        // i값이 바뀜에 따라, 최소한의 원의 영역이 지정한 tracking_ball_size 보다 클 경우, 경계를 그리도록 한다
        for( size_t i = 0; i < contours_r.size(); i++ ){
                    int j=0;
                    int w=0;
                    while ( j < contours_r.size()){
                        if (i==j){
                            j++;
                        }
                        // 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
                        if ( sqrt((center_r[i].x-center_r[j].x)*(center_r[i].x-center_r[j].x)+(center_r[i].y-center_r[j].y)*(center_r[i].y-center_r[j].y)) < (radius_r[i] + radius_r[j]) && radius_r[i] < radius_r[j]){
                            j++;
                            w++;}
                        else {
                            j++;}
                    }
                    if ( w==0 && radius_r[i] > iMin_tracking_ball_size){
                        f++;
                        Scalar color = Scalar( 0, 0, 255);
			if (loop%2=1){
                            for(size_t k1=0; k1<sizeof(center_r1)/sizeof(Point2f); k1++){
				for(size_t k2=0; k2<sizeof(center_r2)/sizeof(Point2f); k2++){
                                if( sqrt((center_r1[k1].x-center_r[i].x)*(center_r1[k1].x-center_r[i].x)+(center_r1[k1].y-center_r[i].y)*(center_r1[k1].y-center_r[i].y))<20 && sqrt((center_r2[k2].x-center_r[i].x)*(center_r2[k2].x-center_r[i].x)+(center_r2[k2].y-center_r[i].y)*(center_r2[k2].y-center_r[i].y))<20){
                                    // 이미지에 경계를 그리고 공의 위치를 정의
                                    drawContours( hsv_frame_red_canny , contours_r_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
                                    vector<float> ball_position_r;
                                    vector<float> ball_position_r1;
                                    vector<float> ball_position_r2;
                                    // 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
                                    ball_position_r = pixel2point(center_r[i], radius_r[i]);
                                    ball_position_r1 = pixel2point(center_r1[k1],radius_r1[k1]);
                                    ball_position_r2 = pixel2point(center_r2[k2],radius_r2[k2]);
                                    float isx = ball_position_r[0];
                                    float isy = ball_position_r[1];
                                    float isz = ball_position_r[2];
                                    float isx1 = ball_position_r1[0];
                                    float isy1 = ball_position_r1[1];
                                    float isz1 = ball_position_r1[2];
                                    float isx2 = ball_position_r2[0];
                                    float isy2 = ball_position_r2[1];
                                    float isz2 = ball_position_r2[2];
                                    float isx = isx * 0.6 + isx1 * 0.3 + isx2 * 0.1;
                                    float isy = isy * 0.6 + isy1 * 0.3 + isy2 * 0.1;
                                    float isz = isz * 0.6 + isz1 * 0.3 + isz2 * 0.1;
                                    isx = roundf(isx*1000)/1000;
                                    isy = roundf(isy*1000)/1000;
                                    isz = roundf(isz*1000)/1000;
                                    string sx = floatToString(isx);
                                    string sy = floatToString(isy);
                                    string sz = floatToString(isz);
                                    text = "Red ball:" + sx + "," + sy + "," + sz;
				    r_msg.size=contours_r.size();
				    r_msg.img_x[i]=isx;
				    r_msg.img_y[i]=isy;
                                    // 공의 중심의 좌표를 화면에 나타낸다
                                    putText(result, text, center_r[i],2,1,Scalar(0,0,255),2);
                                    // 원을 그린다
                                    circle( result, center_r[i], (int)radius_r[i], color, 2, 8, 0 );
                                }
                            }
                        center_r2[f] = center_r[i];
                        radius_r2[f] = radius_r[i];
                    }
		if (loop%2=0){
                            for(size_t k1=0; k1<sizeof(center_r1)/sizeof(Point2f); k1++){
				for(size_t k2=0; k2<sizeof(center_r2)/sizeof(Point2f); k2++){
                                if( sqrt((center_r1[k1].x-center_r[i].x)*(center_r1[k1].x-center_r[i].x)+(center_r1[k1].y-center_r[i].y)*(center_r1[k1].y-center_r[i].y))<20 && sqrt((center_r2[k2].x-center_r[i].x)*(center_r2[k2].x-center_r[i].x)+(center_r2[k2].y-center_r[i].y)*(center_r2[k2].y-center_r[i].y))<20){
                                    // 이미지에 경계를 그리고 공의 위치를 정의
                                    drawContours( hsv_frame_red_canny , contours_r_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
                                    vector<float> ball_position_r;
                                    vector<float> ball_position_r1;
                                    vector<float> ball_position_r2;
                                    // 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
                                    ball_position_r = pixel2point(center_r[i], radius_r[i]);
                                    ball_position_r1 = pixel2point(center_r1[k1],radius_r1[k1]);
                                    ball_position_r2 = pixel2point(center_r2[k2],radius_r2[k2]);
                                    float isx = ball_position_r[0];
                                    float isy = ball_position_r[1];
                                    float isz = ball_position_r[2];
                                    float isx1 = ball_position_r1[0];
                                    float isy1 = ball_position_r1[1];
                                    float isz1 = ball_position_r1[2];
                                    float isx2 = ball_position_r2[0];
                                    float isy2 = ball_position_r2[1];
                                    float isz2 = ball_position_r2[2];
                                    float isx = isx * 0.6 + isx2 * 0.3 + isx1 * 0.1;
                                    float isy = isy * 0.6 + isy2 * 0.3 + isy1 * 0.1;
                                    float isz = isz * 0.6 + isz2 * 0.3 + isz1 * 0.1;
                                    isx = roundf(isx*1000)/1000;
                                    isy = roundf(isy*1000)/1000;
                                    isz = roundf(isz*1000)/1000;
                                    string sx = floatToString(isx);
                                    string sy = floatToString(isy);
                                    string sz = floatToString(isz);
                                    text = "Red ball:" + sx + "," + sy + "," + sz;
				    r_msg.size=contours_r.size();
				    r_msg.img_x[i]=isx;
				    r_msg.img_y[i]=isy;
                                    // 공의 중심의 좌표를 화면에 나타낸다
                                    putText(result, text, center_r[i],2,1,Scalar(0,0,255),2);
                                    // 원을 그린다
                                    circle( result, center_r[i], (int)radius_r[i], color, 2, 8, 0 );
                                }
                            }
                        center_r1[f] = center_r[i];
                        radius_r1[f] = radius_r[i];
                    }

        }
}
		for( size_t i = 0; i < contours_b.size(); i++ ){
			    int j=0;
			    int w=0;
			    while ( j < contours_b.size()){
				if (i==j){
				    j++;
				}
				// 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
				if ( sqrt((center_b[i].x-center_b[j].x)*(center_b[i].x-center_b[j].x)+(center_b[i].y-center_b[j].y)*(center_b[i].y-center_b[j].y)) < (radius_b[i] + radius_b[j]) && radius_b[i] < radius_b[j]){
				    j++;
				    w++;}
				else {
				    j++;}
			    }
			    if ( w==0 && radius_b[i] > iMin_tracking_ball_size){
				Scalar color = Scalar( 255, 0, 0);
				// 이미지에 경계를 그리고 공의 위치를 정의
				drawContours( hsv_frame_blue_canny, contours_b_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
				vector<float> ball_position_b;
				// 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
				ball_position_b = pixel2point(center_b[i], radius_b[i]);
				float isx = ball_position_b[0];
				float isy = ball_position_b[1];
				float isz = ball_position_b[2];
				string sx = floatToString(isx);
				string sy = floatToString(isy);
				string sz = floatToString(isz);
				text = "Blue ball:" + sx + "," + sy + "," + sz;
				// 공의 중심의 좌표를 화면에 나타낸다
				b_msg.size=contours_b.size();
				b_msg.img_x[i]=isx;
				b_msg.img_y[i]=isy;
				putText(result, text, center_b[i],2,1,Scalar(255,0,0),2);
				// 원을 그린다
				circle( result, center_b[i], (int)radius_b[i], color, 2, 8, 0 );
			    }
		}
		int q=-1;
		for( size_t i = 0; i < contours_g.size(); i++ ){
			    int j=0;
			    int w=0;
			    while ( j < contours_g.size()){
				if (i==j){
				    j++;
				}
				// 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
				if ( sqrt((center_g[i].x-center_g[j].x)*(center_g[i].x-center_g[j].x)+(center_g[i].y-center_g[j].y)*(center_g[i].y-center_g[j].y)) < (radius_g[i] + radius_g[j]) && radius_g[i] < radius_g[j]){
				    j++;
				    w++;}
				else {
				    j++;}
			    }
			    if ( w==0 && radius_g[i] > 0.5){
				q++;
				Scalar color = Scalar( 0, 255, 0);
				// 이미지에 경계를 그리고 공의 위치를 정의
				drawContours( hsv_frame_green_canny, contours_g_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
				vector<float> ball_position_g;
				// 공의 중심을 점으로 변환하고 공의 위치를 좌표계로 나타낸다
				ball_position_g = pixel2point(center_g[i], radius_g[i]);
				float isx = ball_position_g[0];
				float isy = ball_position_g[1];
				float isz = ball_position_g[2];
				string sx = floatToString(isx);
				string sy = floatToString(isy);
				string sz = floatToString(isz);
				text = "Green ball:" + sx + "," + sy + "," + sz;
				// 공의 중심의 좌표를 화면에 나타낸다
				g_msg.size=z;
				g_msg.img_x[q]=isx;
				g_msg.img_y[q]=isy;
				putText(result, text, center_g[i],2,1,Scalar(0,255,0),2);
				// 원을 그린다
				circle( result, center_g[i], (int)radius_g[i], color, 2, 8, 0 );
			    }
		}
        // 원래 이미지와 경계를 그린 이미지, 프로세싱한 이미지를 보여준다
        // imshow("Video Capture",calibrated_frame);
        // imshow("Object Detection_HSV_Blue",hsv_frame_blue);
        // imshow("Object Detection_HSV_Green",hsv_frame_green);
        // imshow("Object Detection_YCbCr_Red",resultYCB_r);
        // imshow("Canny Edge for Blue Ball", hsv_frame_blue_canny);
        // imshow("Canny Edge for Red Ball", hsv_frame_red_canny);
        // imshow("Canny Edge for Green Ball", hsv_frame_green_canny);
        // imshow("Result", result);




	//cv::imshow("view", frame);  //show the image with a window
	cv::waitKey(1); //main purpose of this command is to wait for a key command. However, many highgui related functions like imshow, redraw, and resize can not work without this command...just use it!
	red_pub.publish(r_msg);  //publish a message
	blue_pub.publish(b_msg);
	green_pub.publish(g_msg);

}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{

	if(msg -> height==480&&buffer.size().width==320){
	std::cout<<"resized"<<std::endl;
   //check the size of the image received. if the image have 640x480, then change the buffer size to 640x480.
	cv::resize(buffer,buffer,cv::Size(640,480));
}
   else{
	//do nothing!
   }

   try
   {
     buffer = cv_bridge::toCvShare(msg, "bgr8")->image;  //transfer the image data into buffer
   }
   catch (cv_bridge::Exception& e)
   {
     ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
   }
   ball_detect(); //proceed ball detection
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "ball_detect_node"); //init ros nodd
	ros::NodeHandle nh; //create node handler
	image_transport::ImageTransport it(nh); //create image transport and connect it to node hnalder
	image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback); //create subscriber

	//setting publisher
	blue_pub = nh.advertise<core_msgs::ball_position>("/b_position", 100);
	red_pub = nh.advertise<core_msgs::ball_position>("/r_position", 100);
	green_pub = nh.advertise<core_msgs::ball_position>("/g_position",100);
	// pub_markers = nh.advertise<visualization_msgs::Marker>("/balls",1);

	ros::spin(); //spin.
	return 0;
}

// int를 string으로 변환하는 함수
string intToString(int n)
{
    stringstream s;
    s << n;
    return s.str();
}

// string을 float으로 변환하는 함수
string floatToString(float f)
{
    ostringstream buffer;
    buffer << f;
    return buffer.str();
}

//Standard Dilate and erode functions to improve white/black areas in Binary Image
//Pointer &thresh used so it affects threshImg so it can be used in tracking.
void morphOps(Mat &thresh){

    //create structuring element that will be used to "dilate" and "erode" image.
    //the element chosen here is a 3px by 3px rectangle
    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

	// 경계 사이로 이미지를 erode 한다
    erode(thresh,thresh,erodeElement);
    erode(thresh,thresh,erodeElement);

	// 경계 사이로 이미지를 dilate 한다
    dilate(thresh,thresh,dilateElement);
    dilate(thresh,thresh,dilateElement);
}

//pixel 값을 position 값으로 변환하는 함수 정의
vector<float> pixel2point(Point center, int radius){

	// 변수 정의
    vector<float> position;
    float x, y, u, v, Xc, Yc, Zc;

	// 중심의 x,y 좌표 정의
	x = center.x;
    y = center.y;

	//  새로운 x,y 값 계산
	u = (x-intrinsic_data[2])/intrinsic_data[0];
    v = (y-intrinsic_data[5])/intrinsic_data[4];

	// 반지름을 고려하여 x,y,z 값 계산
    Zc = (intrinsic_data[0]*fball_radius)/(2*(float)radius) ;
    Xc = u*Zc ;
    Yc = v*Zc ;

	// 소수 셋째 자리까지만 나타냄
    Xc = roundf(Xc * 1000) / 1000;
    Yc = roundf(Yc * 1000) / 1000;
    Zc = roundf(Zc * 1000) / 1000;

	// Xc, Yc, Zc를 position matrix 의 가장 밑에 위치시킴
    position.push_back(Xc);
    position.push_back(Yc);
    position.push_back(Zc);

    return position;
}

void on_low_y_thresh_trackbar_red(int, void *)
{
low_y_r = min(high_y_r-1, low_y_r);
setTrackbarPos("Low Y","Object Detection_YCbCr_Red", low_y_r);
}
void on_high_y_thresh_trackbar_red(int, void *)
{
high_y_r = max(high_y_r, low_y_r+1);
setTrackbarPos("High Y", "Object Detection_YCbCr_Red", high_y_r);
}
void on_low_cr_thresh_trackbar_red(int, void *)
{
low_cr_r = min(high_cr_r-1, low_cr_r);
setTrackbarPos("Low Cr","Object Detection_YCbCr_Red", low_cr_r);
}
void on_high_cr_thresh_trackbar_red(int, void *)
{
high_cr_r = max(high_cr_r, low_cr_r+1);
setTrackbarPos("High Cr", "Object Detection_YCbCr_Red", high_cr_r);
}
void on_low_cb_thresh_trackbar_red(int, void *)
{
low_cb_r= min(high_cb_r-1, low_cb_r);
setTrackbarPos("Low Cb","Object Detection_YCbCr_Red", low_cb_r);
}
void on_high_cb_thresh_trackbar_red(int, void *)
{
high_cb_r = max(high_cb_r, low_cb_r+1);
setTrackbarPos("High Cb", "Object Detection_YCbCr_Red", high_cb_r);
}

//파란색 인식을 위한 track bar 생성하는 함수 생성
void on_low_h_thresh_trackbar_blue(int, void *)
{
low_h_b = min(high_h_b-1, low_h_b);
setTrackbarPos("Low H","Object Detection_HSV_Blue", low_h_b);
}
void on_high_h_thresh_trackbar_blue(int, void *)
{
high_h_b = max(high_h_b, low_h_b+1);
setTrackbarPos("High H", "Object Detection_HSV_Blue", high_h_b);
}
void on_low_s_thresh_trackbar_blue(int, void *)
{
low_s_b = min(high_s_b-1, low_s_b);
setTrackbarPos("Low S","Object Detection_HSV_Blue", low_s_b);
}
void on_high_s_thresh_trackbar_blue(int, void *)
{
high_s_b = max(high_s_b, low_s_b+1);
setTrackbarPos("High S", "Object Detection_HSV_Blue", high_s_b);
}
void on_low_v_thresh_trackbar_blue(int, void *)
{
low_v_b= min(high_v_b-1, low_v_b);
setTrackbarPos("Low V","Object Detection_HSV_Blue", low_v_b);
}
void on_high_v_thresh_trackbar_blue(int, void *)
{
high_v_b = max(high_v_b, low_v_b+1);
setTrackbarPos("High V", "Object Detection_HSV_Blue", high_v_b);
}

void on_low_h_thresh_trackbar_green(int, void *)
{
low_h_g = min(high_h_g-1, low_h_g);
setTrackbarPos("Low H","Object Detection_HSV_Green", low_h_g);
}
void on_high_h_thresh_trackbar_green(int, void *)
{
high_h_g = max(high_h_g, low_h_g+1);
setTrackbarPos("High H", "Object Detection_HSV_Green", high_h_g);
}
void on_low_s_thresh_trackbar_green(int, void *)
{
low_s_g = min(high_s_g-1, low_s_g);
setTrackbarPos("Low S","Object Detection_HSV_Green", low_s_g);
}
void on_high_s_thresh_trackbar_green(int, void *)
{
high_s_g = max(high_s_g, low_s_g+1);
setTrackbarPos("High S", "Object Detection_HSV_Green", high_s_g);
}
void on_low_v_thresh_trackbar_green(int, void *)
{
low_v_g = min(high_v_g-1, low_v_g);
setTrackbarPos("Low V","Object Detection_HSV_Green", low_v_g);
}
void on_high_v_thresh_trackbar_green(int, void *)
{
high_v_g = max(high_v_g, low_v_g+1);
setTrackbarPos("High V", "Object Detection_HSV_Green", high_v_g);
}
//빨간색과 파란색을 인식하면 edge를 그리도록 하는 함수 생성
void on_canny_edge_trackbar_red(int, void *)
{
setTrackbarPos("Min Threshold", "Canny Edge for Red Ball", lowThreshold_r);
}
void on_canny_edge_trackbar_blue(int, void *)
{
setTrackbarPos("Min Threshold", "Canny Edge for Blue Ball", lowThreshold_b);
}
void on_canny_edge_trackbar_green(int, void *)
{
setTrackbarPos("Min Threshold", "Canny Edge for Green Ball", lowThreshold_g);
}
