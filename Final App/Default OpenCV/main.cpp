//
//  main.cpp
//  HIST-348
//
//  Created by Jonah Joselow on 12/9/13.
//  Copyright (c) /Users/jonahjoselow/Dropbox/Georgetown/2013 - Fall/HIST 348 - Art Science Tech in the Renaissance/Research Project/Code/OpenCV 1/OpenCV 1.xcodeproj2013 Jonah Joselow. All rights reserved.
//
//  Main driver app
//

#include <CoreGraphics/CoreGraphics.h> // used to get screen size
#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "ImageHolder.h"

using namespace std;
using namespace cv;


ImageHolder imageToAnalyze;


bool dragged = false; // true if dragging

void onMouseAnalyzeColor(int event, int x, int y, int flags, void* param){
    switch( event ){
        case CV_EVENT_MOUSEMOVE:
            if (dragged) {
                //cout << "MOVED" << endl;
                imageToAnalyze.clearColorBox();
                int width = x - imageToAnalyze.boxToAnalyzeColor.x;
                int height = y - imageToAnalyze.boxToAnalyzeColor.y;
                
                imageToAnalyze.setColorRectangleSize(width, height);
                imageToAnalyze.drawColorRectangle();
            }
            break;
        case CV_EVENT_LBUTTONDOWN:  //start drawing
            
            dragged = true;
            imageToAnalyze.setColorRectangleStart(x, y);
            //cout << "CLICKED" << endl;
            break;
        case CV_EVENT_LBUTTONDBLCLK:  //double click to clear
            break;
        case CV_EVENT_LBUTTONUP:  //draw what we created with Lbuttondown
            dragged = false;
            
            break;
    }
}


void onMouseDrawPerspective(int event, int x, int y, int flags, void* param)
{
    //if (currentAppPhase == DRAW_PERSPECTIVE) {
    switch( event ){
        case CV_EVENT_MOUSEMOVE:
            if (dragged) {
                imageToAnalyze.redrawLines();
                //cout << "MOVED" << endl;
                Scalar color;
                if (currentDrawingMode == DRAWING_PERSPECTIVE)
                    color = Scalar(0, 0, 255);
                else if (currentDrawingMode == DRAWING_HORIZON)
                    color = Scalar(0, 200, 0);
                
                line(imageToAnalyze.image_perspective_drawn, imageToAnalyze.line_start, cv::Point(x,y), color, 3, 8);
                imshow(imageToAnalyze.nameOfWindow+ " - Draw Lines", imageToAnalyze.image_perspective_drawn);
            }
            break;
        case CV_EVENT_LBUTTONDOWN:  //start drawing
            imageToAnalyze.continuingToDrawLines(true);
            imageToAnalyze.setLineStart(x, y);
            dragged = true;
            //cout << "CLICKED" << endl;
            break;
        case CV_EVENT_LBUTTONDBLCLK:  //double click to clear
            break;
        case CV_EVENT_LBUTTONUP:  //draw what we created with Lbuttondown
            dragged = false;
            imageToAnalyze.continuingToDrawLines(false);
            
            //cout << "RELESAED" << endl;
            imageToAnalyze.setLineEnd(x, y);
            if (currentDrawingMode == DRAWING_PERSPECTIVE){
                imageToAnalyze.drawLine();
            }
            else if (currentDrawingMode == DRAWING_HORIZON){
                currentHorizonLineDrawn = true;
                imageToAnalyze.drawLine();
            }
            
            break;
    }
    //}
    
    
}

int main(int argc, const char * argv[])
{
    CGRect mainMonitor = CGDisplayBounds(CGMainDisplayID());
    CGFloat monitorHeight = CGRectGetHeight(mainMonitor) - 100;
    CGFloat monitorWidth = CGRectGetWidth(mainMonitor);
    
    currentAppPhase = SELECT_IMAGE;
    
    // get File
    //string nameOfFile = "renaissance-3.jpg";
    string nameOfFile = "Masaccio7_tribute money.jpg";
    Mat sourceImage = imread(nameOfFile, 1);
    if (!sourceImage.data) { // check if file can be opened
        cout << " | ERROR: unable to read image: " << nameOfFile << endl;
        return -1;
    }
    
    try {cvtColor(sourceImage, sourceImage, CV_RGB2RGBA);}catch(...){} // convert color to CV_RGB2RGBA
    
    //
    // adjust size to be perfect for display
    //
    double imageRatio = (double) sourceImage.cols / (double) sourceImage.rows;
    double ratioWidth = monitorWidth / sourceImage.cols;
    double ratioHeight = monitorHeight / sourceImage.rows;
    
    double numTimesLarger = min(ratioWidth, ratioHeight);
    
    char shouldResize;
    
    cout << "This image can become " << numTimesLarger << " time larger than is is now" << endl;
    
    if (numTimesLarger < 1.0) {
        cout << nameOfFile << " is larger than this monitor. In order to" << endl;
        cout << "properly analyze this image, it will be automatically adjusted" << endl;
        cout << "This ratio " << imageRatio << " and colors will be preserved in this transformation." << endl;
        shouldResize = 'y';
    }
    else {
        cout << nameOfFile << " is smaller than this monitor" << endl;
        cout << "Would you like it to be automatically adjusted?" << endl;
        cout << "This ratio " << imageRatio << " and colors will be preserved in this transformation." << endl;
        cout << "y / n : ";
        cin >> shouldResize;
    }
    if (shouldResize == 'y' || shouldResize == 'Y') {
        resize(sourceImage, sourceImage, cv::Size(cv::Point(sourceImage.cols * numTimesLarger, sourceImage.rows * numTimesLarger)), 0, 0, INTER_AREA);
    }
    
    imageToAnalyze.image_original = sourceImage.clone();
    imageToAnalyze.nameOfWindow = nameOfFile;
    cout << "Opened " << nameOfFile << " successfully." << endl;
    
    currentAppPhase = SET_PREFERENCES;
    
    //    char shouldPathDetectionBeEnabled;
    //    cout << "Would you like to use path detection? (y / n): ";
    //    cin >> shouldPathDetectionBeEnabled;
    //
    //    if (shouldPathDetectionBeEnabled == 'y' || shouldPathDetectionBeEnabled == 'Y') { imageToAnalyze.enablePathDetection(); }
    
    currentAppPhase = DRAW_PERSPECTIVE;
    cout << "Now begin tracing lines toward the center of the image." << endl;
    cout << "Left-click and drag your cursor until satisfied" << endl;
    cout << "Keys: " << endl;
    cout << "   D : delete last line drawn" << endl;
    cout << "   Q : quit program" << endl;
    cout << "   P : print perspective lines" << endl;
    cout << "   R : redraw perspective lines" << endl;
    cout << "   C : Calculate vanishing point and horizon" << endl;
    cout << "   N : Move on to color analysis" << endl;
    
    namedWindow(imageToAnalyze.nameOfWindow + " - Draw Lines", WINDOW_AUTOSIZE);
    imageToAnalyze.drawPerspective();
    
    while(cvWaitKey(0) == 'd' || cvWaitKey(0) == 'D') { // delete key
        imageToAnalyze.deleteLastLine();
    }
    
    
    while(true) {
        switch (cvWaitKey(0)) {
            case'r':
                imageToAnalyze.drawPerspective();
                cout << "REDRAWN" << endl;
                break;
            case'R':
                imageToAnalyze.drawPerspective();
                cout << "REDRAWN" << endl;
                break;
            case 'c':
                imageToAnalyze.calculateVanishingPoint();
                break;
            case 'C':
                imageToAnalyze.calculateVanishingPoint();
                break;
            case 'p':
                imageToAnalyze.printPerspectiveLines();
                break;
            case 'P':
                imageToAnalyze.printPerspectiveLines();
                break;
            case 't':
                imageToAnalyze.togglePathDetection();
                break;
            case 'T':
                imageToAnalyze.togglePathDetection();
                break;
            case 'd':
                imageToAnalyze.deleteLastLine();
                break;
            case 'D':
                imageToAnalyze.deleteLastLine();
                break;
            case 'n':
                currentAppPhase = CALC_COLOR;
                cout << "Please draw rectangle around atmosphere. " << endl;
                imageToAnalyze.analyzeColor();
                break;
            case 'N':
                currentAppPhase = CALC_COLOR;
                cout << "Please draw rectangle around atmosphere. " << endl;
                imageToAnalyze.analyzeColor();
                break;
            case 'F':
                imageToAnalyze.detectFaces();
                break;
            case 'f':
                imageToAnalyze.detectFaces();
                break;
            case 'q':
                return 0;
            case 'Q':
                return 0;
            default:
                break;
        }
    }
    
    while (cvWaitKey(0) != 27);
    
    
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}

