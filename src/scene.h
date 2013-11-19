// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef SCENE_H
#define SCENE_H

#include "glm/glm.hpp"
#include "utilities.h"
#include <vector>
#include "sceneStructs.h"
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

class scene{
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadObject(string objectid);
    int loadCamera();
public:
    scene(string filename);
    ~scene();

    vector<geom> objects;
    vector<material> materials;
    camera renderCam;
};

#endif
