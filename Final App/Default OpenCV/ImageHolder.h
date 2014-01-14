//
//  ImageHolder.h
//  OpenCV 1
//
//  Created by Jonah Joselow on 12/9/13.
//  Copyright (c) 2013 Jonah Joselow. All rights reserved.
//
//  Class to hold image analysis
//

#ifndef OpenCV_1_ImageHolder_h
#define OpenCV_1_ImageHolder_h

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Line.h"

using namespace cv;
using namespace std;

enum appPhase {SELECT_IMAGE, SET_PREFERENCES, DRAW_PERSPECTIVE, CALC_COLOR};
enum drawingMode {DRAWING_PERSPECTIVE, DRAWING_HORIZON};
bool currentHorizonLineDrawn = false;

drawingMode currentDrawingMode = DRAWING_PERSPECTIVE;
appPhase currentAppPhase;

void onMouseDrawPerspective(int event, int x, int y, int flags, void* param);
void onMouseAnalyzeColor(int event, int x, int y, int flags, void* param);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
std::string face_cascade_name = "haarcascade_frontalface_alt.xml";
std::string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

class ImageHolder {
private:
    int numberOfTimesVanishingPointHasBeenCalculated = 0; // num times vanishing point calculated
    std::vector<cv::Point> allVanishingPoints; // holds all calculated vanishing points
    Line horizonLine = Line(cv::Point(0,0), cv::Point(0,0)); // hold horizonLine
    
    std::vector<Line> allPerspectiveLines;   // list of all lines drawn (up and down)
    bool pathDetectionEnabled = false; // true if path detection is enabled
    bool drawingPerspectiveLine = false; // true if currently drawing lines
    double radiusOfEllipse = 0;
    
    
    void setColorRectangleStart( int x, int y) {
        boxToAnalyzeColor = cv::Rect(x, y, 0, 0);
    }
    
    void setColorRectangleSize(int width, int height) {
        boxToAnalyzeColor.width = width;
        boxToAnalyzeColor.height = height;
    }
    
    void drawColorRectangle() {
        
        
        if(boxToAnalyzeColor.width > 0 && boxToAnalyzeColor.height > 0){ // if the width and height are above 0
            // draw rectangle on top of image_color_analysis to show area that is being sampled
            cv::rectangle(image_color_analysis,
                          cv::Point(boxToAnalyzeColor.x, boxToAnalyzeColor.y),
                          cv::Point(boxToAnalyzeColor.x + boxToAnalyzeColor.width, boxToAnalyzeColor.y + boxToAnalyzeColor.height),
                          cv::Scalar(0, 165, 255),
                          2,
                          8
                          );
            imshow(nameOfWindow + " - Color Analysis", image_color_analysis);
            
            Mat croppedImage;// = image_color_analysis(boxToAnalyzeColor);
            Mat(image_original, boxToAnalyzeColor).copyTo(croppedImage); // crop from original image
            
            /// Separate the image in 3 places ( B, G and R )
            vector<Mat> bgr_planes;
            split( croppedImage, bgr_planes );
            
            /// Establish the number of bins
            int histSize = 256;
            
            /// Set the ranges ( for B,G,R) )
            float range[] = { 0, 256 } ;
            const float* histRange = { range };
            
            bool uniform = true, accumulate = false;
            
            Mat b_hist, g_hist, r_hist;
            
            /// Compute the histograms:
            calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate ); // B
            calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate ); // G
            calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate ); // R
            
            // Draw the histograms for B, G and R
            int hist_w = 512, hist_h = 400;
            int bin_w = cvRound( (double) hist_w / histSize );
            
            Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
            
            /// Normalize the result to [ 0, histImage.rows ]
            normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() ); // B
            normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() ); // G
            normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() ); // R
            
            /// Draw for each channel
            for( int i = 1; i < histSize; i++ ) {
                line(
                     histImage,
                     cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                     cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                     Scalar( 255, 0, 0),
                     2, 8, 0  );
                line(
                     histImage,
                     cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                     cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                     Scalar( 0, 255, 0),
                     2, 8, 0  );
                line(
                     histImage,
                     cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                     cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                     Scalar( 0, 0, 255),
                     2, 8, 0  );
            }
            
            imshow(nameOfWindow + " - Color Sample", histImage);
            
        }
        else {
            cout << "Rectangle has negative width or height." << endl;
            cout << "Width : " << boxToAnalyzeColor.width << endl;
            cout << "Height: " << boxToAnalyzeColor.height << endl;
            cout << "X     : " << boxToAnalyzeColor.x << endl;
            cout << "Y     : " << boxToAnalyzeColor.y << endl;
            
            int startX = boxToAnalyzeColor.x;
            int startY = boxToAnalyzeColor.y;
            int width = boxToAnalyzeColor.width;
            int height = boxToAnalyzeColor.height;
            
            int newWidth = 0;
            int newHeight = 0;
            int newStartX = 0;
            int newStartY = 0;
            
            cv::Point newStart;
            cv::Rect drawingRect;
            if ( (width > startX) && (height > startY)) { // +w, +h
                // no adjust needed
            }
            else if ( (width > startX) && (height < startY)) { // +w, -h
                
            }
            else if ( (width < startX) && (height < startY)) { // -w, -h
                newStartX = startX + width;
                newStartY = startY + height;
                newStart = cv::Point(width, height);
                newWidth = abs(width);
                newHeight = abs(height);
                
                drawingRect = cvRect(newStartX, newStartY, newWidth, newHeight);
                
                
                cv::rectangle(image_color_analysis,
                              cv::Point(drawingRect.x, drawingRect.y),
                              cv::Point(startX, startY),
                              cv::Scalar(0, 165, 255),
                              2,
                              8
                              );
                imshow(nameOfWindow + " - Color Analysis", image_color_analysis);
                
            }
            else if ( (width < startX) && (height > startY)) { // -w, +h
                
            }
            
            
            
            try{
                destroyWindow(nameOfWindow + " - Color Sample");
            }catch(...){}
        }
        
    }
    
    void drawLine() {
        Line thisLine(line_start, line_end);
        
        Scalar color;
        if (currentDrawingMode == DRAWING_PERSPECTIVE){
            color = Scalar(0, 0, 255);
            allPerspectiveLines.push_back(thisLine);
            line(image_perspective_drawn, thisLine.start, thisLine.end, color, 3, 8);
        }
        else if (currentDrawingMode == DRAWING_HORIZON) {
            horizonLine = thisLine;
            drawHorizonLine();
        }
//        cout << "Drawing Line" << endl;
//        cout << "start: " << thisLine.start << endl;
//        cout << "end:   " << thisLine.end << endl;
        
        imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
    }
    
    
protected:
    cv::Point line_start; // holds start for line
    cv::Point line_end;   // holds end for line
public:
    cv::Rect boxToAnalyzeColor;
    void deleteLastLine() {
        if (allPerspectiveLines.size() > 0){
            allPerspectiveLines.pop_back();
            redrawLines();
            imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
        }
        else {
            cout << " | ERROR: no lines drawn" << endl;
        }
        
    }
    
    void clearColorBox() {
        image_color_analysis = image_original.clone();
    }

    
    void redrawColorBox() {
        image_color_analysis = image_original.clone();
        
        drawColorRectangle();
    }
    void redrawLines() {
        if (pathDetectionEnabled){
            image_perspective_drawn = image_original.clone();
            enablePathDetection();
        }
        else
            image_perspective_drawn = image_original.clone();
        
        if (allPerspectiveLines.size() > 0) {
            for (int i = 0; i < allPerspectiveLines.size(); i++) {
                line(image_perspective_drawn, allPerspectiveLines[i].start, allPerspectiveLines[i].end, Scalar( 0, 0, 255 ), 3, 2); // Perspective Lines
            }
        }
        
    }
    
    void continuingToDrawLines(bool truth) { drawingPerspectiveLine = truth;}
    
    void setLineStart(int x, int y) { line_start = cv::Point(x, y); }
    void setLineEnd(int x, int y) { line_end = cv::Point(x, y); }
    
    string nameOfWindow;
    
    cv::Mat image_original;
    cv::Mat image_perspective_drawn;
    cv::Mat image_color_analysis;
    
    void togglePathDetection() {
        image_perspective_drawn = image_original.clone();
        pathDetectionEnabled = !pathDetectionEnabled;
        
        if (pathDetectionEnabled)enablePathDetection();
        
        redrawLines();
        try {imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);}
        catch(...) {
            image_perspective_drawn = image_original.clone();
            imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
        }
    }
    
    
    void detectFaces(){
        //destroyAllWindows();
        if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); cout << "ERROR" << endl; return; };
        if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); cout << "ERROR" << endl; return; };
        //image_face_detection = image_original.clone();
        
        
        std::vector<cv::Rect> faces;
        cv::Mat frame_gray;
        
        cvtColor( image_original, frame_gray, CV_RGB2GRAY );
        equalizeHist( frame_gray, frame_gray );
        
        //-- Detect faces
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
        
        for( size_t i = 0; i < faces.size(); i++ )
        {
            cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
            ellipse( image_perspective_drawn, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
            
            Mat faceROI = frame_gray( faces[i] );
            std::vector<cv::Rect> eyes;
            
            //-- In each face, detect eyes
            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
            
            for( size_t j = 0; j < eyes.size(); j++ )
            {
                cv::Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( image_perspective_drawn, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
            }
        }
        //-- Show what you got
        imshow( nameOfWindow + " - Draw Lines", image_perspective_drawn );
    }
    
    void enablePathDetection() {
        image_perspective_drawn = image_original.clone();
        Mat src_gray;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        
        
        GaussianBlur( image_perspective_drawn, image_perspective_drawn, cv::Size(3,3), 0, 0, BORDER_DEFAULT );
        
        /// Convert it to gray
        cvtColor( image_perspective_drawn, src_gray, CV_RGB2GRAY );
        
        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
    
        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        
        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        
        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image_perspective_drawn );
        cvtColor(image_perspective_drawn, image_perspective_drawn, CV_GRAY2RGBA);
    }
    
    friend void onMouseDrawPerspective(int event, int x, int y, int flags, void* param);
    friend void onMouseAnalyzeColor(int event, int x, int y, int flags, void* param);
    
    void drawPerspective() {
        drawingPerspectiveLine = true;
        currentDrawingMode = DRAWING_PERSPECTIVE;
        if (pathDetectionEnabled)
            image_perspective_drawn = image_perspective_drawn.clone();
        else
            image_perspective_drawn = image_original.clone();
        
        redrawLines();
        
        imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
        setMouseCallback(nameOfWindow  + " - Draw Lines", onMouseDrawPerspective);
    }
    
    void analyzeColor() {
        image_color_analysis = image_original.clone();
        
        destroyWindow(nameOfWindow + " - Draw Lines");
        
        
        imshow(nameOfWindow + " - Color Analysis", image_color_analysis);
        setMouseCallback(nameOfWindow  + " - Color Analysis", onMouseAnalyzeColor);
        
    }
    
    void printPerspectiveLines() {
        cout << endl;
        cout << "Printing all lines." << endl;
        for (int i = 0; i < allPerspectiveLines.size(); i++) {
            cout << "Line: " << i << " : " << allPerspectiveLines[i].to_string() << endl;
        }
    }
    
    cv::Point calculateIntersection(Line lineOne, Line lineTwo){
        //cout << lineOne.printEquation() << endl;
        //cout << lineTwo.printEquation() << endl;
        
        double compositeXCoefficient = lineOne.m - lineTwo.m; // m1*x - m2*x
        double compositeYIntercept = lineTwo.b - lineOne.b; // b2 - b1
        
        double xCoord = compositeYIntercept / compositeXCoefficient;
        
        double yCoord = lineOne.calculateYGivenX(xCoord);
        
        cv::Point thisPoint(xCoord, yCoord);
        return thisPoint;
    }
    
    // Method Citation
    // http://snipplr.com/view/40484/
    void kMeans(vector<cv::Point> &datavector, const int clusterCount,
                vector < vector<cv::Point> > &clusterContainer)
    {
        /*
         *  Pre:  "datavector" the data to be clustered by K-Means
         *        "clusterCount" how many clusters you want
         *
         *  Post: "classContainer" I pack the points with the same cluster into vector, so it
         *        is a vetor of vector
         */
        
        
        int dataLength = datavector.size();
        
        
        // Put data into suitable container
        CvMat* points = cvCreateMat(dataLength, 1, CV_32FC2);
        CvMat* clusters = cvCreateMat(dataLength, 1, CV_32SC1 );
        
        for (int row = 0; row < points->rows; row++) {
            float* ptr = (float*)(points->data.ptr + row*points->step);
            for (int col = 0; col < points->cols; col++) {
                *ptr = static_cast<float>(datavector[row].x);
                ptr++;
                *ptr = static_cast<float>(datavector[row].y);
            }
        }
        
        // The Kmeans algorithm function (OpenCV function)
        cvKMeans2(points, clusterCount, clusters, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1, 2));
        
        // Pack result to 'classContainer', each element in 'classContainer' means one cluster,
        // each cluster is one vector<CvPoint> contain all points belong to this cluster
        int clusterNum;
        vector<cv::Point> tempClass;
        
        for (int i = 0; i < clusterCount; i++) {
            tempClass.clear();
            
            for (int row = 0; row < clusters->rows; row++) {
                
                
                float* p_point = (float*)(points->data.ptr + row*points->step);
                int X = static_cast<int>(*p_point) ;
                p_point++;
                int Y = static_cast<int>(*p_point);
                
                clusterNum = clusters->data.i[row];
                
                if (clusterNum == i)
                    tempClass.push_back(cvPoint(X, Y));
                
            }
            
            clusterContainer.push_back(tempClass);
            
        }
        
        // Remove empty cluster
        for (vector< vector<cv::Point> >::size_type i = 0; i < clusterContainer.size(); ++i) {
            
            bool isEmpty = clusterContainer[i].empty();
            
            if (isEmpty) {           
                vector< vector<cv::Point> >::iterator iter = clusterContainer.begin();
                iter = iter + i;
                clusterContainer.erase(iter);   
                i = i - 1;
            }
        }   
        
        cvReleaseMat(&points);
        cvReleaseMat(&clusters);
        
        
    }
    
    
    vector<cv::Point> calculateVanishingPoint() { // vector because there can be more than one
        
        vector<cv::Point> pointsOfIntersection;
        for(int indexOfEachLine = 0; indexOfEachLine < allPerspectiveLines.size();indexOfEachLine++){
            
            for (int indexOfAllOtherLines = 0; indexOfAllOtherLines < allPerspectiveLines.size(); indexOfAllOtherLines++) {
                if (indexOfEachLine != indexOfAllOtherLines) {
                    Line firstLine = allPerspectiveLines[indexOfEachLine];
                    Line secondLine = allPerspectiveLines[indexOfAllOtherLines];
                    
                    cv::Point IntersectionPoint = calculateIntersection( firstLine, secondLine);
                    
                    
                    
                    if ( (IntersectionPoint.x != -2147483648) && (IntersectionPoint.y != -2147483648) )// if not infinity
                    {
                        pointsOfIntersection.push_back(IntersectionPoint);
                        
//                        cout << "Lines: " << endl;
//                        cout << " | " << firstLine.to_string() << endl;
//                        cout << " | " << secondLine.to_string() << endl;
//                        cout << " > " << IntersectionPoint.x << " , " << IntersectionPoint.y << endl;
//                        circle(image_perspective_drawn, IntersectionPoint, 25, Scalar( 255, 0, 0 ), 5, 8, 0);
//                        imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
                    }
                    
                    
                    
                }
            }
        }
        
        
        //cout << "Possible Points:" << endl;
        for(int i = 0; i < pointsOfIntersection.size(); i++){
            //cout << "point " << i << " : " << pointsOfIntersection[i].x << " , " << pointsOfIntersection[i].y << endl;
            
            //circle(image_perspective_drawn, pointsOfIntersection[i], 5, Scalar( 255, 0, 0 ), 5, 8, 0);
            //imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
        }
        
        
        vector<vector<cv::Point>> clusteredPoints;
        vector<vector<cv::Point>> onlyPositiveClusteredPoints;
        vector<cv::Point> onlyPositivePoints;
        
        
        if (pointsOfIntersection.size() >= 3) {
            int numberOfClusters = 2;
            if (pointsOfIntersection.size() / 4 > 3) {
                numberOfClusters = pointsOfIntersection.size() / 4;
            }
            kMeans(pointsOfIntersection, 3, clusteredPoints);
            allVanishingPoints.clear();
            for(int i = 0; i < clusteredPoints.size(); i++){
                
                int AverageX = 0, AverageY = 0;
                int numInserted = 0;
                
                for (int c = 0; c < clusteredPoints[i].size(); c++) {
                    //cout << "point " << i << " : " << clusteredPoints[i][c].x << " , " << clusteredPoints[i][c].y << endl;
                    
                    if ((clusteredPoints[i][c].x > 0) && (clusteredPoints[i][c].y > 0)) {
                        AverageX += clusteredPoints[i][c].x;
                        AverageY += clusteredPoints[i][c].y;
                        onlyPositivePoints.push_back(clusteredPoints[i][c]);
                        numInserted++;
                        
//                        circle(image_perspective_drawn, clusteredPoints[i][c], 25, Scalar( 255, 0, 0 ), 5, 8, 0);
//                        imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
                    
                    
                        
                        cv::Point averagePointFromCluster(clusteredPoints[i][c].x, clusteredPoints[i][c].y);
                        allVanishingPoints.push_back(averagePointFromCluster);
                        
                        circle(image_perspective_drawn, averagePointFromCluster, 10, Scalar( 255, 0, 0 ), 5, 8, 0);
                    }
                    
                    
                    
                    
                }
                
                // Further Clustering if needed
//                if (onlyPositivePoints.size() > 3) {
//                    kMeans(onlyPositivePoints, 3, onlyPositiveClusteredPoints);
//
//                    int PositiveAverageX = 0, PositiveAverageY = 0;
//
//                    for (int v = 0; v < onlyPositiveClusteredPoints.size(); v++) {
//                        for (int q = 0; q < onlyPositiveClusteredPoints[v].size(); q++) {
//                            PositiveAverageX += onlyPositiveClusteredPoints[v][q].x;
//                            PositiveAverageY += onlyPositiveClusteredPoints[v][q].y;
//                        }
//
//                        PositiveAverageX /= onlyPositiveClusteredPoints[v].size();
//                        PositiveAverageY /= onlyPositiveClusteredPoints[v].size();
//                        circle(image_perspective_drawn, cv::Point(PositiveAverageX, PositiveAverageY), 25, Scalar( 255, 0, 0 ), 5, 8, 0);
//                    }
//                }
//                else{
//                    for (int v = 0; v < onlyPositivePoints.size(); v++) {
//                        circle(image_perspective_drawn, onlyPositivePoints[v], 25, Scalar( 255, 0, 0 ), 5, 8, 0);
//                    }
//                }
                if (numInserted > 0) {
                    AverageX /= numInserted;
                    AverageY /= numInserted;
                    //allVanishingPoints.clear();
                    
                    cv::Point averagePointFromCluster(AverageX, AverageY);
                    allVanishingPoints.push_back(averagePointFromCluster);
                    
                    circle(image_perspective_drawn, averagePointFromCluster, 10, Scalar( 255, 0, 0 ), 5, 8, 0);
                    //cout << "Draw: " << averagePointFromCluster.x << " , " << averagePointFromCluster.y << endl;
                }
                imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
            }

        }
        else{
            cout << " | ERROR: please draw " << 3 - pointsOfIntersection.size() << " more lines." << endl;
            imshow(nameOfWindow + " - Draw Lines", image_perspective_drawn);
        }
        
        
        if (!currentHorizonLineDrawn) {
            cout << "Please draw a horizontal line to the painting" << endl;
            currentDrawingMode = DRAWING_HORIZON;
        }
        else{
            drawHorizonLine();
            drawElipse();
        }
        
        return pointsOfIntersection;
    }
    
    void drawElipse() {
        int AverageX = 0;
        int AverageY = 0;
        for (int eachPoint = 0; eachPoint < allVanishingPoints.size(); eachPoint++) {
            AverageY += allVanishingPoints.at(eachPoint).y;
            AverageX += allVanishingPoints.at(eachPoint).x;
        }
        
        AverageY /= allVanishingPoints.size();
        AverageX /= allVanishingPoints.size();
        
        int LargestDistanceAwayFromAverage = 0;
        Vector<double> allDistances; // holds all distances
        for (int eachPoint = 0; eachPoint < allVanishingPoints.size(); eachPoint++) {
            
            // d = abs( sqrt ( (x2 - x1)^2 + (y2 - y1)^2))
            // c^2 = a^2 + b^2
            double  x2minusx1 = allVanishingPoints.at(eachPoint).x - AverageX;
            double xGroupSquared = pow(x2minusx1, 2);
            
            double  y2minusy1 = allVanishingPoints.at(eachPoint).y - AverageY;
            double yGroupSquared = pow(y2minusy1, 2);
            
            double localDistance = abs(sqrt(xGroupSquared + yGroupSquared));
            
//            cout << "Distance between : " << AverageX << " , " << AverageY << endl;
//            cout << "               & : " << allVanishingPoints.at(eachPoint).x << " , " << allVanishingPoints.at(eachPoint).y << endl;
//            cout << "                 = " << localDistance << endl;
//            cout << endl;
            
            if (localDistance > LargestDistanceAwayFromAverage) {
                LargestDistanceAwayFromAverage = localDistance;
            }
            
            allDistances.push_back(localDistance);
        }
        
        std::sort(allDistances.begin(), allDistances.end(), std::greater<double>()); // sort desc order
        
//        cout << "largest = " << allDistances[0] << endl;
//        cout << "2nd lar = " << allDistances[1] << endl;
        
        radiusOfEllipse = allDistances[1];
        cv::Scalar gold = cv::Scalar(0, 215, 255);
        
        cout << "Majority of clusters can be captured by a circle with radius = " << radiusOfEllipse << endl;
        ellipse(image_perspective_drawn,
               cv::Point(AverageX, AverageY),
               cv::Size(radiusOfEllipse, radiusOfEllipse),
               0,
               0,
               360,
               gold,
               2,
               8);
    }
    

    void drawHorizonLine() {
        
        redrawLines();
        
        Scalar color = Scalar(0, 200, 0);
        
        int AverageY = 0;
        int AverageX = 0;
        
        for (int i = 0; i < allVanishingPoints.size(); i++) {
            AverageY += allVanishingPoints.at(i).y;
            AverageX += allVanishingPoints.at(i).x;
        }
        
        AverageY /= allVanishingPoints.size();
        AverageX /= allVanishingPoints.size();
        
        
        //cout << "Horizon Line: " << horizonLine.to_string() << endl;
        
        int x0, x1, x2, y0, y1, y2;
        
        
        double YDifference = horizonLine.calculateYGivenX(AverageX) - AverageY;
        
        
        horizonLine.b -= YDifference; // move to level.
        x0 = 0;
        y0 = horizonLine.calculateYGivenX(x0);
        x1 = image_perspective_drawn.cols;
        y1 = horizonLine.calculateYGivenX(x1);
        
        cv::Point X0(x0, y0);
        cv::Point X1(x1, y1);
        
//        horizonLine.start = newStartPoint;
//        horizonLine.end = newEndPoint;
        
        
        //cout << "NEW Line: " << horizonLine.to_string() << endl;
        horizonLine.start = X0;
        horizonLine.end = X1;
        

        line(image_perspective_drawn, horizonLine.start, horizonLine.end, color, 3, 2);
        
        currentDrawingMode = DRAWING_PERSPECTIVE;
        
    }
};




#endif
