// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef RAYTRACEKERNEL_H
#define PATHTRACEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "KDTreeStructs.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

extern glm::vec3* accumulatorImage;

void cudaAllocateAccumulatorImage(camera *renderCam);
void cudaFreeAccumulatorImage();
void cudaClearAccumulatorImage(camera *renderCam);
void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, 
					  material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, cameraData liveCamera);

//photon mapping
void cudaPhotonMapCore(camera* renderCam, int frame, int iterations, uchar4* PBOPos, cameraData liveCamera);

void initTexture(cudatexture* textures, float4* cputexturedata, int numberOfTextures, int widthcount, int maxheight);
//for allocating and deallocating memory
void cudaAllocateMemory(int targetFrame, camera* renderCam, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,
						glm::vec3* vertices, int numberOfVertices, glm::vec3* normals, int numberOfNormals, triangle* faces, int numberOfFaces,
						glm::vec2* uvs, int numberOfUVs, KDNodeGPU* kdTree, int numberOfNodes, int rootIndex, int* primIndex, int numberOfPrimIndices);
void cudaFreeMemory();
void cudaFreeTexture();

int streamCompactPhotons (photon* inputPhotons, photon* outputPhotons, int size);

__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y);


//for cpu debugging
void cudaDrawCPUImage(uchar4 * pos, camera* renderCam, glm::vec3* cpuImage);

#endif
