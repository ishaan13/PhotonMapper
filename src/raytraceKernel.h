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

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define PHOTONMAP 1

extern glm::vec3* accumulatorImage;

void cudaAllocateAccumulatorImage(camera *renderCam);
void cudaFreeAccumulatorImage();
void cudaClearAccumulatorImage(camera *renderCam);
void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, 
					  material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, cameraData liveCamera);

#if PHOTONMAP
//photon mapping
void cudaPhotonMapCore(camera* renderCam, int frame, int iterations, uchar4* PBOPos);

//for allocating and deallocating memory
void cudaAllocateMemory(camera* renderCam, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms);
void cudaFreeMemory();

__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y);
#endif

#endif
