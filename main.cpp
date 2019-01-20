#include "HogGetter.h"
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// parameters
// #define NEG_SET
#define POS_SET

#define IMG_COLS 32
#define IMG_ROWS 32

#define RAW_SAMPLE_FLIP_PARAM 1 // 翻转参数 
#define RAW_SAMPLE_PATH "D:/baseRelate/code/svm_trial/BackUpSource/Ball/Train/Raw/"
#define RAW_SAMPLE_POSTFIX "/*.jpg"

#define SAVE_PATH "D:/baseRelate/code/svm_trial/BackUpSource/Ball/Train/Neg/"
#define SAVE_POSTFIX ".jpg"

#define NEG_SAMPLE_CUT_TIMES (7)

#define COUNTER_INIT_NUM 0

bool selection = false;
bool drawing_box = false;
cv::Rect box;
cv::Mat raw_image;
cv::Mat proc_image;

int counter = COUNTER_INIT_NUM;


string GetNextPath() {
    stringstream t_ss;
    string t_s;

    t_ss << counter++;
    t_ss >> t_s;
    t_s = SAVE_PATH + t_s;
    t_s += SAVE_POSTFIX;
    cout<<t_s<<endl;
    return t_s;
}

void mouseHandler(int event, int x, int y, int flags, void *param) {
    switch (event) {
    case CV_EVENT_RBUTTONDBLCLK:
        selection = false;
        drawing_box = false;
        cv::imshow("233", raw_image);
    case CV_EVENT_MOUSEMOVE:
        if (drawing_box) {
            box.width = x - box.x;
            box.height = (IMG_ROWS / IMG_COLS)*box.width;

            cv::Mat for_show;
            for_show = raw_image.clone();
            cv::rectangle(for_show, box, cv::Scalar(255, 0, 0));
            cv::imshow("233", for_show);
        }
        break;
    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = cv::Rect(x, y, 0, 0);
        break;
    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        if (box.width < 0) {
            box.x += box.width;
            box.width *= -1;
        }
        if (box.height < 0) {
            box.y += box.height;
            box.height *= -1;
        }
        selection = true;
        proc_image = raw_image(box);

        cv::resize(proc_image, proc_image, cv::Size(IMG_COLS, IMG_ROWS));
        cv::imwrite(GetNextPath(), proc_image);

        cv::flip(proc_image, proc_image, RAW_SAMPLE_FLIP_PARAM);
        cv::imwrite(GetNextPath(), proc_image);
        break; 
    }
}


int main(int argc, char const *argv[]) {
    HogGetter hog_getter;
    cout<<"here"<<endl;
    hog_getter.ImageReader_(RAW_SAMPLE_PATH, RAW_SAMPLE_POSTFIX);
    
    // auto images = hog_getter.raw_images_;

    cv::namedWindow("233");
    cv::setMouseCallback("233", mouseHandler, 0);


    for (auto i = hog_getter.raw_images_.begin(); i != hog_getter.raw_images_.end(); i++) {
        cout<<"here"<<endl;
#ifdef POS_SET
        while (i->cols > 1500 || i->rows > 900) {
            cv::resize((*i), (*i), cv::Size((*i).cols/10, (*i).rows/10));
        }
        raw_image = i->clone();
        cv::imshow("233", *i);
        while(1) {
            char key = cv::waitKey(50);
            if (key == 'r') {
                (*i) = i->t();
                raw_image = i->clone();
                cv::imshow("233", *i);
            }
            else if(key == 'n') {
                break;
            }
        }
#endif
#ifdef NEG_SET
        std::vector<cv::Mat> t_neg_samples;
        hog_getter.set_window_size(cv::Size(IMG_COLS*4, IMG_ROWS*4));
        for (int j = 0; j < NEG_SAMPLE_CUT_TIMES; j++) {
            cv::Mat t_result = hog_getter.RandomCutter_(*i);
            cv::resize(t_result, t_result, cv::Size(IMG_COLS, IMG_ROWS));
            t_neg_samples.push_back(t_result);
            // t_neg_samples.push_back(hog_getter.RandomCutter_(*i));
        }
        for (auto j = t_neg_samples.begin(); j != t_neg_samples.end(); j++) {
            cv::imwrite(GetNextPath(), *j);
            cv::flip(*j, *j, RAW_SAMPLE_FLIP_PARAM);
            cv::imwrite(GetNextPath(), *j);
        } 
#endif

    }
    return 0;
}
