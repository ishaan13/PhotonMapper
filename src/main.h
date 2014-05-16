// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef MAIN_H
#define MAIN_H

#ifdef __APPLE__
	#include <GL/glfw.h>
#else
	#include <GL/glew.h>
	#include <GL/glut.h>
#endif

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include <ctime>

#include "glslUtility.h"
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "image.h"
#include "raytraceKernel.h"
#include "utilities.h"
#include "scene.h"
#include "KDTree.h"

#if CUDA_VERSION >= 5000
    #include <helper_cuda.h>
    #include <helper_cuda_gl.h>
    #define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
    #include <cutil_inline.h>
    #include <cutil_gl_inline.h>
    #define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif


#define CPUTRACE 0
#define OUTPUT_DATA 0

using namespace std;

//-------------------------------
//----------PATHTRACER-----------
//-------------------------------

scene* renderScene;
camera* renderCam;
int targetFrame;
int previousFrame;
int iterations;
bool finishedRender;
bool singleFrameMode;
bool firstEverExecution = true;

//-------------------------------
//--------PHOTON MAPPER----------
//-------------------------------
geom* geoms;
material* materials;
triangle* faces;
glm::vec3* vertices;
glm::vec3* normals;
glm::vec2* uvs;

int numberOfGeoms;
int numberOfMaterials;
int numberOfVertices;
int numberOfNormals;
int numberOfFaces;
int numberOfTextures;
int numberOfUVs;

//-------------------------------
//------------KD TREE -----------
//-------------------------------
KDTree* kdTree;

// for cpu
glm::vec3* cpuImage;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
cameraData liveCamera;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1280; int height=720;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();
void sendCurrentFrameDataToGPU();

#ifdef __APPLE__
	void display();
#else
	void display();
	void keyboard(unsigned char key, int x, int y);
	void specialKeyboard(int key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

enum {
	DISP_RAYTRACE,
	DISP_PHOTONS,
	DISP_GATHER,
	DISP_COMBINED,
	DISP_PATHTRACE,
	DISP_KDHEAT,
	DISP_TOTAL
};

enum {
	KD_ON,
	KD_OFF
};

enum {
	VIS_LIN,
	VIS_HSV,
	VIS_NUM
};

int mode = DISP_COMBINED;
int kdmode = KD_ON;
int visMode = VIS_LIN;


#ifdef __APPLE__
	void init();
#else
	void init(int argc, char* argv[]);
#endif

void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath);

void initKDTree();

void copyDataFromScene();
void sendCurrentFrameDataToGPU();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void freeCPUMemory();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);



////CPU TESTING////
void cpuRaytrace();

#endif
