// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <iostream>
#include "image.h"
#include "stb_image/stb_image_write.h"
#include "utilities.h"

image::image(int x, int y){
    xSize = x;
    ySize = y;
    redChannel   = new float[x*y];
    greenChannel = new float[x*y];
    blueChannel  = new float[x*y];
    alphaChannel = new float[x*y];
    for(int i=0; i<(x*y); i++){
        redChannel[i]   = 0;
        greenChannel[i] = 0;
        blueChannel[i]  = 0;
        alphaChannel[i] = 0;
    }
    gamma.applyGamma=false;
    gamma.divisor=1;
    gamma.gamma=1;
}

image::~image(){
    delete redChannel;
    delete greenChannel;
    delete blueChannel;
    delete alphaChannel;
}

//------------------------
//----Image Operations----
//------------------------

float image::applyGamma(float f){
    //apply gamma correction, use simple power law gamma for now. TODO: sRGB
    return pow(f/float(gamma.divisor), gamma.gamma);
}

void image::saveImageRGB(string filename){
    unsigned char* bitmapData = new unsigned char[3 * xSize * ySize];
    int i = 0;
    
    //read data to buffer for stb_image output
    for(int y = 0; y < ySize; y++) {
        for(int x = 0; x < xSize; x++) { 
            if(gamma.applyGamma){
                bitmapData[i]   = (unsigned char)utilityCore::clamp(applyGamma(readPixelR(x,y))*255,0,255);
                bitmapData[i+1] = (unsigned char)utilityCore::clamp(applyGamma(readPixelG(x,y))*255,0,255);
                bitmapData[i+2] = (unsigned char)utilityCore::clamp(applyGamma(readPixelB(x,y))*255,0,255);
            }else{
                bitmapData[i]   = (unsigned char)utilityCore::clamp(readPixelR(x,y)*255, 0, 255);
                bitmapData[i+1] = (unsigned char)utilityCore::clamp(readPixelG(x,y)*255, 0, 255);
                bitmapData[i+2] = (unsigned char)utilityCore::clamp(readPixelB(x,y)*255, 0, 255);
                
            }
            i=i+3;
        }
    }
    
    //check requested output type
    int imagetype = 0; //0 for png, 1 for bmp
    
    if(filename[filename.size()-1]=='\r'){
        //OSX Version
        if(filename[filename.size()-4]=='b' && filename[filename.size()-3]=='m' && filename[filename.size()-2]=='p'){
            imagetype = 1;
        }
    }else{
        //Windows Version
        if(filename[filename.size()-3]=='b' && filename[filename.size()-2]=='m' && filename[filename.size()-1]=='p'){
            imagetype = 1;
        }
    }
    
    //write output file
    if(imagetype==1){
        stbi_write_bmp(filename.c_str(), xSize, ySize, 3, bitmapData);
    }else{
        stbi_write_png(filename.c_str(), xSize, ySize, 3, bitmapData, xSize * 3);
    }
}

//------------------------
//-------GETTERS----------
//------------------------

glm::vec3 image::readPixelRGB(int x, int y){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        return glm::vec3(redChannel[index], greenChannel[index], blueChannel[index]);
    }else{
        return glm::vec3(0,0,0);
    }
}

glm::vec4 image::readPixelRGBA(int x, int y){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        return glm::vec4(redChannel[index], greenChannel[index], blueChannel[index], alphaChannel[index]);
    }else{
        return glm::vec4(0,0,0,0);
    }
}

float image::readPixelR(int x, int y){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        return redChannel[index];
    }else{
        return 0;
    }
}

float image::readPixelG(int x, int y){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        return greenChannel[index];
    }else{
        return 0;
    }
}

float image::readPixelB(int x, int y){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        return blueChannel[index];
    }else{
        return 0;
    }
}

float image::readPixelA(int x, int y){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        return alphaChannel[index];
    }else{
        return 0;
    }
}

float* image::getRedChannel(){
    return redChannel;
}

float* image::getGreenChannel(){
    return greenChannel;
}

float* image::getBlueChannel(){
    return blueChannel;
}

float* image::getAlphaChannel(){
    return alphaChannel;
}

glm::vec3* image::getRGBChannels(){
    glm::vec3* rgb = new glm::vec3[xSize*ySize];
    for(int i=0; i<(xSize*ySize); i++){
        rgb[i] = glm::vec3(redChannel[i], greenChannel[i], blueChannel[i]);
    }
    return rgb;
}

glm::vec4* image::getRGBAChannels(){
    glm::vec4* rgb = new glm::vec4[xSize*ySize];
    for(int i=0; i<(xSize*ySize); i++){
        rgb[i] = glm::vec4(redChannel[i], greenChannel[i], blueChannel[i], alphaChannel[i]);
    }
    return rgb;
}

glm::vec2 image::getDimensions(){
    return glm::vec2(xSize, ySize);
}

gammaSettings image::getGammaSettings(){
    return gamma;
}

//------------------------
//-------SETTERS----------
//------------------------

void image::writePixelRGB(int x, int y, glm::vec3 pixel){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        redChannel[index] = pixel[0];
        greenChannel[index] = pixel[1];
        blueChannel[index] = pixel[2];
    }
}

void image::writePixelRGBA(int x, int y, glm::vec4 pixel){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        redChannel[index] = pixel[0];
        greenChannel[index] = pixel[1];
        blueChannel[index] = pixel[2];
        alphaChannel[index] = pixel[3];
    }
}

void image::writePixelR(int x, int y, float pixel){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        redChannel[index] = pixel;
    }
}

void image::writePixelG(int x, int y, float pixel){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        greenChannel[index] = pixel;
    }
}

void image::writePixelB(int x, int y, float pixel){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        blueChannel[index] = pixel;
    }
}

void image::writePixelA(int x, int y, float pixel){
    if(!(x<0 || y<0 || x>=xSize || y>=ySize)){
        int index = (y*xSize)+x;
        alphaChannel[index] = pixel;
    }
}

void image::setGammaSettings(gammaSettings newGamma){
    gamma = newGamma;
}
