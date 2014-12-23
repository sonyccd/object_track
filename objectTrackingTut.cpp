#include <sstream>
#include <iostream>
#include <opencv2/core/internal.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
int MAX_NUM_OBJECTS = 10;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT * FRAME_WIDTH / 1.5;
const string windowName = "RAW";
const string windowName1 = "HSV";
const string windowName2 = "Thresholded Image";
const string trackbarWindowName = "Trackbars";
bool trackObjects = true;
bool useMorphOps = true;
bool showContours = true;
int mouseX=0;
int mouseY=0;
int H_step=256;
int S_step=256;
int V_step=256;
CvRect box;
bool drawing_box=false;
Mat cameraFeed;

void on_trackbar(int, void *) {}

string intToString(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void draw_box( Mat &img){
    Scalar color(rand() & 255, rand() & 255, rand() & 255);
    rectangle( img, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),
            color );
}

Scalar find_mean_colors(Mat img){
    Scalar roi;
    try {
        Mat temp = Mat(img, box);
        roi = mean(temp);
    }catch(int e){
        cout<<"ERROR "<<e<<endl;
    }
    return roi;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    switch( event ){
        case CV_EVENT_MOUSEMOVE:
            if( drawing_box ){
                box.width = x-box.x;
                box.height = y-box.y;
            }
            break;

        case CV_EVENT_LBUTTONDOWN:
            drawing_box = true;
            box = cvRect( x, y, 0, 0 );
            break;

        case CV_EVENT_LBUTTONUP:
            drawing_box = false;
            if( box.width < 0 ){
                box.x += box.width;
                box.width *= -1;
            }
            if( box.height < 0 ){
                box.y += box.height;
                box.height *= -1;
            }
            break;

        case CV_EVENT_RBUTTONDOWN:
            box.height=0;
            box.width=0;
            box.x=0;
            box.y=0;
            break;
    }
}

void createTrackbars() {
    namedWindow(trackbarWindowName, 0);
    createTrackbar("H_MAX", trackbarWindowName, &H_step, H_step, on_trackbar);
    createTrackbar("S_MAX", trackbarWindowName, &S_step, S_step, on_trackbar);
    createTrackbar("V_MAX", trackbarWindowName, &V_step, V_step, on_trackbar);
    createTrackbar("Obj Max",trackbarWindowName, &MAX_NUM_OBJECTS,MAX_NUM_OBJECTS,on_trackbar);
}

void drawTarget(int x, int y, Mat &frame) {
    circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);

    if (y - 25 > 0) {
        line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
    }else {
        line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
    }

    if (y + 25 < FRAME_HEIGHT) {
        line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
    }else {
        line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
    }

    if (x - 25 > 0) {
        line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
    }else {
        line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
    }

    if (x + 25 < FRAME_WIDTH) {
        line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
    }else {
        line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);
    }

    putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
}

void morphOps(Mat &thresh) {
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
    erode(thresh, thresh, erodeElement);
    erode(thresh, thresh, erodeElement);
    dilate(thresh, thresh, dilateElement);
    dilate(thresh, thresh, dilateElement);
}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {
    Mat temp;
    threshold.copyTo(temp);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    double refArea = 0;
    bool objectFound = false;
    if(showContours) {
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(cameraFeed, contours, -1, color);
    }
    if (hierarchy.size() > 0) {
        int numObjects = hierarchy.size();
        //std::cout<<"Number of objects: "<<numObjects<<std::endl;
        if (numObjects < MAX_NUM_OBJECTS) {
            for (int index = 0; index >= 0; index = hierarchy[index][0]) {
                Moments moment = moments((cv::Mat) contours[index]);
                double area = moment.m00;
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                if (area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA && area > refArea) {
                    x = moment.m10 / area;
                    y = moment.m01 / area;
                    objectFound = true;
                    refArea = area;
                } else {
                    objectFound = false;
                }
            }
            if (objectFound == true) {
                putText(cameraFeed, "Tracking...", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
                drawTarget(x, y, cameraFeed);
            }
        } else {
            putText(cameraFeed, "TOO MUCH NOISE", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
        }
    }
}

int main(int argc, char *argv[]) {
    Mat HSV;
    Mat threshold;
    int x = 0, y = 0;
    createTrackbars();
    VideoCapture capture;
    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    while (true) {
        capture.read(cameraFeed);
        cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
        setMouseCallback("RAW", CallBackFunc,NULL);
        //filter HSV image between values and store filtered image to
        //threshold matrix
        Scalar temp= find_mean_colors(HSV);
        H_MIN=(int)temp[0];
        S_MIN=(int)temp[1];
        V_MIN=(int)temp[2];
        H_MAX=(int)temp[0]+H_step;
        S_MAX=(int)temp[1]+S_step;
        V_MAX=(int)temp[2]+V_step;
        inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
        if (useMorphOps) {
            morphOps(threshold);
        }
        //pass in thresholded frame to our object tracking function
        //this function will return the x and y coordinates of the
        //filtered object
        if (trackObjects) {
            trackFilteredObject(x, y, threshold, cameraFeed);
        }
        //show frames
        draw_box(cameraFeed);
        imshow(windowName2, threshold);
        imshow(windowName1, HSV);
        imshow(windowName, cameraFeed);
        if(waitKey(30)==27){
            break;
        }
    }
    return 0;
}

