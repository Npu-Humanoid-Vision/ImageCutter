#include "HogGetter.h"
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

bool selection = false;
bool drawing_box = false;
cv::Rect box;
cv::Mat raw_image;
cv::Mat raw_image_backup;
cv::Mat proc_image;

int counter = 0;
string save_path = "D:/baseRelate/code/svm_trial/BackUpSource/People/Test/HegSample/";

void mouseHandler(int event, int x, int y, int flags, void *param) {
    switch (event) {
    case CV_EVENT_MOUSEMOVE:
        if (drawing_box) {
            box.width = x - box.x;
            box.height = 2*box.width;

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
        proc_image = raw_image_backup(box);

        stringstream t_ss;
        string t_s;

        t_ss << counter++;
        t_ss >> t_s;

        t_s = save_path + t_s;
        t_s += ".jpg";

        cv::resize(proc_image, proc_image, cv::Size(64, 128));
        cv::imwrite(t_s, proc_image);
        break;
    }
}


int main(int argc, char const *argv[]) {
    HogGetter hog_getter;

    hog_getter.ImageReader_("D:/78things/INRIAPerson/INRIAPerson/Test/neg/", "*.png");
    
    auto images = hog_getter.raw_images_;

    cv::namedWindow("233");
    cv::setMouseCallback("233", mouseHandler, 0);
    for (auto i = images.begin(); i != images.end(); i++) {
        // cv::resize((*i), (*i), cv::Size((*i).cols/2, (*i).rows/2));

        // raw_image = i->clone();
        // raw_image_backup = i->clone();
        // cv::imshow("233", *i);
        // // cv::resizeWindow("233", (*i).cols/2, (*i).rows/2);

        // while (cv::waitKey(20) != 'n') {
        //     cv::waitKey(20);
        // }
        (*i) = hog_getter.RandomCutter_(*i); 

        stringstream t_ss;
        string t_s;

        t_ss << counter++;
        t_ss >> t_s;

        t_s = save_path + t_s;
        t_s += ".jpg";

        cv::imwrite(t_s, *i);
    }
    return 0;
}
