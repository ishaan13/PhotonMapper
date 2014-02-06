// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "KDTreeStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations

__host__ __device__ unsigned int hash(unsigned int a);
__host__ __device__ bool epsilonCheck(float a, float b);

__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec4 multiplyMV_4(cudaMat4 m, glm::vec4 v);
__host__ __device__ cudaMat4 getNormalTransform(cudaMat4 a);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal,glm::vec2& uv);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv);
__host__ __device__ float triangleIntersectionTest(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3,
													glm::vec2 t1, glm::vec2 t2, glm::vec2 t3, ray r, glm::vec3& intersection, glm::vec3& normal, glm::vec2& uv);
__host__ __device__ void getRandomPointNormalUVOnCube(staticGeom cube, float randomSeed, glm::vec3& point, glm::vec3& normal, glm::vec2& uv);
__host__ __device__ void getRandomPointNormalUVOnSphere(staticGeom cube, float randomSeed, glm::vec3& point, glm::vec3& normal, glm::vec2& uv);

__host__ __device__ glm::vec3 getNormalOfPointOnUnitCube(glm::vec3 point);
__host__ __device__ glm::vec2 getUVOfPointOnUnitCube(glm::vec3 point);
__device__ float getClosestIntersection(ray r, staticGeom* geoms, int numberOfGeoms, triangle* faces, int numberOfFaces, glm::vec3* vertices,
												glm::vec3* normals, glm::vec2* uvs, glm::vec3& minIntersectionPoint, glm::vec3& minNormal,
												int& intersectedGeom, int& intersectedMaterial, glm::vec2& minUV, KDNodeGPU* cudakdtree, int treeRootIndex, int* cudaPrimIndex,
												int kdmode);

__device__ bool visibilityCheck(ray r, staticGeom* geoms, int numberOfGeoms, triangle* faces, int numberOfFaces, glm::vec3* vertices,
								glm::vec3* normals, glm::vec2* uvs, glm::vec3 pointToCheck, int lightSourceIndex, KDNodeGPU* cudakdtree, int treeRootIndex, int* cudaPrimIndex,
								int kdmode);

__host__ __device__ float planeIntersectionTest(glm::vec3 pointOnPlane, glm::vec3 normalOfPlane, ray r, glm::vec3 &intersection);


__host__ __device__ void buildAABB(staticGeom& geom);

__host__ __device__ cudaMat4 glmMat4ToCudaMat4(glm::mat4 a);

__host__ __device__ glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);


#endif