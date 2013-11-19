// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef Raytracer_image_h
#define Raytracer_image_h

#include "glm/glm.hpp"

using namespace std;

struct gammaSettings{
    bool applyGamma;
    int divisor;
    float gamma;
};

class image{
private:
    float* redChannel;
    float* greenChannel;
    float* blueChannel;
    float* alphaChannel;
    int xSize;
    int ySize;
    gammaSettings gamma;
public:
    image(int x, int y);
    ~image();
    
    //------------------------
    //-------GETTERS----------
    //------------------------
    glm::vec3 readPixelRGB(int x, int y);
    glm::vec4 readPixelRGBA(int x, int y);
    float readPixelR(int x, int y);
    float readPixelG(int x, int y);
    float readPixelB(int x, int y);
    float readPixelA(int x, int y);
    float* getRedChannel();
    float* getBlueChannel();
    float* getGreenChannel();
    float* getAlphaChannel();
    glm::vec3* getRGBChannels();
    glm::vec4* getRGBAChannels();
    glm::vec2 getDimensions();
    gammaSettings getGammaSettings();
    
    //------------------------
    //-------SETTERS----------
    //------------------------
    void writePixelRGB(int x, int y, glm::vec3 pixel);
    void writePixelRGBA(int x, int y, glm::vec4 pixel);
    void writePixelR(int x, int y, float pixel);
    void writePixelG(int x, int y, float pixel);
    void writePixelB(int x, int y, float pixel);
    void writePixelA(int x, int y, float pixel);
    void setGammaSettings(gammaSettings newGamma);
    
    //------------------------
    //----Image Operations----
    //------------------------
    void saveImageRGB(string filename);
    float applyGamma(float f);
    
};


#endif
