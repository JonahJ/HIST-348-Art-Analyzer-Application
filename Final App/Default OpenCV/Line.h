//
//  Line.h
//  OpenCV 1
//
//  Created by Jonah Joselow on 12/9/13.
//  Copyright (c) 2013 Jonah Joselow. All rights reserved.
//
//  Class to hold perspective lines
//

#ifndef OpenCV_1_Line_h
#define OpenCV_1_Line_h

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

class Line{
public:
    cv::Point start;
    cv::Point end;
    double m, b; // y = mx + b
    
    Line(cv::Point s, cv::Point e) {
        start = s;
        end = e;
        
        calculateEquationOfLine();
    }
    
    void calculateSlope() {
        
        double numerator;
        double denominator;
        if (start.x > end.x) {
            numerator = start.y - end.y;
            denominator = start.x - end.x;
        }
        else{
            numerator = end.y - start.y;
            denominator = end.x - start.x;
        }
        m = numerator / denominator;
    }
    
    void calculateYIntercept() {
        double y_term = start.y; // y
        double x_term = start.x * m; // m * x
        b = y_term - x_term;
    }
    
    double calculateYGivenX(double _x) {
        double xAnswer = _x * m; // m * x
        double yAnswer = xAnswer + b; // m*x + b
        return yAnswer;
    }
    
    double calculateXGivenY(double _y) {
        int yAnswer = _y - b; // y - b
        yAnswer = yAnswer / m; // (y - b) / m
        return yAnswer;
    }
    
    double calculateAverageX() {
        double average = start.x + end.x;
        average /= 2;
        return average;
    }
    
    double calculateAverageY() {
        double average = start.y + end.y;
        average /= 2;
        return average;
    }
    
    std::string to_string() {
        std::string equation;
        
        equation = "y = ";
        
        std::string _m;
        std::ostringstream convert;   // stream used for the conversion
        convert << m;
        _m = convert.str();
        
        equation += _m;
        equation += " * x + ";
        
        std::string _b;
        std::ostringstream convert2;   // stream used for the conversion
        convert2 << b;
        _b = convert2.str();
        
        equation += _b;
        
        return equation;
    }
    
    void calculateEquationOfLine() {
        calculateSlope();
        calculateYIntercept();
    }
};

#endif
