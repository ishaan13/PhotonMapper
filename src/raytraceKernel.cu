// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

// Optimizations and add ons
#define JITTER 1
#define COMPACTION 1
#define ACCUMULATION 1
#define DOF 1
#define FRESNEL 1
#define SCHLICK 0
#define PATHTRACER 0	
#define PAINTERLY 0
#define PHOTONMAP 1

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

enum {
	DISP_RAYTRACE,
	DISP_PHOTONS,
	DISP_GATHER,
	DISP_COMBINED,
	DISP_PATHTRACE,
	DISP_TOTAL
};


#if PHOTONMAP
int numPhotons = 10000;

int numBounces = 5;			//hard limit of 3 bounces for now
float totalEnergy = 80;			//total amount of energy in the scene, used for calculating flux per photon


photon* cudaPhotonPool;		//global variable of photons
glm::vec3* cudaPhotonMapImage;

#define RADIUS 1.5

#endif

glm::vec3* accumulatorImage = NULL;
extern bool singleFrameMode;
extern int mode;

//scene data
glm::vec3* cudaimage;
staticGeom* cudageoms;
material* cudamaterials;
int* cudaLights;
float* cudaAccumLightProbabilities;
int numLights;
int numGeoms;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	getchar();
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov
#if DOF
											,float aperture, float focusPlane
#endif
	){
  ray r;

  // @DO: verify field of view!
  glm::vec3 axis_a = glm::cross(view, up);
  glm::vec3 axis_b = glm::cross(axis_a, view);
  glm::vec3 midPoint = eye + view;
  glm::vec3 viewPlaneX = axis_a * tan(PI_F * fov.x/180.0f) * glm::length(view)/glm::length(axis_a);
  glm::vec3 viewPlaneY = axis_b * tan(PI_F * fov.y/180.0f) * glm::length(view)/glm::length(axis_b);

#if JITTER
  glm::vec3 jitter = generateRandomNumberFromThread(resolution,time,x,y);
  glm::vec3 screenPoint = midPoint +
							(2.0f * ((jitter.x + 1.0f * x) / (resolution.x-1)) - 1.0f) * viewPlaneX + 
							(1.0f - 2.0f * ((jitter.y + 1.0f * y) / (resolution.y-1))) * viewPlaneY;
#else
  glm::vec3 screenPoint = midPoint +
							(2.0f * (1.0f * x / (resolution.x-1)) - 1.0f) * viewPlaneX + 
							(1.0f - 2.0f * (1.0f * y / (resolution.y-1))) * viewPlaneY;

#endif

#if DOF

  glm::vec3 focusPlaneIntersection;
  
  r.origin = eye;
  r.direction = glm::normalize(screenPoint - eye);

  glm::vec3 focusPlaneCenter = r.origin + r.direction * focusPlane;
  planeIntersectionTest(focusPlaneCenter,view,r,focusPlaneIntersection);

  glm::vec3 apertureJitter = aperture * (generateRandomNumberFromThread(resolution,time,x,y) - 0.5f);
  r.origin = r.origin + apertureJitter;
  r.direction = glm::normalize(focusPlaneIntersection - r.origin);

#else
  r.origin = screenPoint;
  r.direction = glm::normalize(screenPoint - eye);
#endif
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float frames){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){
      glm::vec3 color;
      color.x = image[index].x*255.0 / frames;
      color.y = image[index].y*255.0 / frames;
      color.z = image[index].z*255.0 / frames;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__device__ bool visibilityCheck(ray r, staticGeom* geoms, int numberOfGeoms, glm::vec3 pointToCheck, int geomShotFrom, int lightSourceIndex)
{
	bool visible = true;
	float distance = glm::length(r.origin - pointToCheck);

	// Check whether any object occludes point to check from ray's origin
	for(int iter=0; iter < numberOfGeoms; iter++)
	{
		// Avoid calculating self intersections
		if(iter==lightSourceIndex)
			continue;

		float depth=-1;
		glm::vec3 intersection;
		glm::vec3 normal;
		
		if(geoms[iter].type == CUBE)
		{
			depth = boxIntersectionTest(geoms[iter],r,intersection,normal);
		}
		
		
		else if(geoms[iter].type == SPHERE)
		{
			depth = sphereIntersectionTest(geoms[iter],r,intersection,normal);
		}
		
		if(depth > 0 && (depth + NUDGE) < distance)
		{
			//printf("Depth: %f\n", depth);
			visible = false;
			break;
		}
	}

	
	return visible;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, ray* rayPool
#if COMPACTION
							, int numberOfRays
#endif
							){
	int index;
	int x,y;
#if COMPACTION
	index = blockIdx.x * blockDim.x + threadIdx.x;
	y = index / resolution.x;
	x = index - y * resolution.x;
	if (index < numberOfRays) {
#else
	x = (blockIdx.x * blockDim.x) + threadIdx.x;
	y = (blockIdx.y * blockDim.y) + threadIdx.y;
	index = x + (y * resolution.x);
	if((x<=resolution.x && y<=resolution.y) && glm::length(rayPool[index].transmission) > FLOAT_EPSILON){
#endif

	ray r = rayPool[index];	


	//Check all geometry for intersection
	int intersectedGeom = -1;
	int intersectedMaterial = -1;
	float minDepth = 1000000.0f;
	glm::vec3 minIntersectionPoint;
	glm::vec3 minNormal = glm::vec3(0.0f);
	for(int iter=0; iter < numberOfGeoms; iter++)
	{
		float depth=-1;
		glm::vec3 intersection;
		glm::vec3 normal;
		staticGeom currentGeometry = geoms[iter];
		if(currentGeometry.type == CUBE)
		{
			depth = boxIntersectionTest(currentGeometry,r,intersection,normal);
		}
		
		else if(geoms[iter].type == SPHERE)
		{
			depth = sphereIntersectionTest(currentGeometry,r,intersection,normal);
		}
		

		if(depth > 0 && depth < minDepth)
		{
			minDepth = depth;
			minIntersectionPoint = intersection;
			minNormal = normal;
			intersectedGeom = iter;
			intersectedMaterial = currentGeometry.materialid;
		}
	}

	// Depth render - test
	//float maxDepth = 15.0f;
	
	glm::vec3 diffuseLight = glm::vec3(0.0f);
	glm::vec3 phongLight = glm::vec3(0.0f);

	glm::vec3 diffuseColor;
	glm::vec3 specularColor;
	glm::vec3 emittance;

	//Calculate Lighting if any geometry is intersected
	if(intersectedGeom > -1)
	{
		//finalColor = materials[geoms[intersectedGeom].materialid].color;
		material m = materials[intersectedMaterial];
		diffuseColor = m.color;
		specularColor = m.specularColor;
		// Emmited color is the same as material color
		emittance = m.color * m.emittance;

#if PATHTRACER != 1
		// Stochastic Diffused Lighting with "area" lights
		for(int iter = 0; iter < numberOfGeoms; iter++)
		{
			material lightMaterial = materials[geoms[iter].materialid];
			// If this geometry is going to act like a light source
			if(lightMaterial.emittance > 0.0001f)
			{
				glm::vec3 lightSourceSample, normal;

				// Get a random point on the light source
				if(geoms[iter].type == SPHERE)
				{
					getRandomPointAndNormalOnSphere(geoms[iter],time*index, lightSourceSample, normal);
				}
				else if(geoms[iter].type == CUBE)
				{
					getRandomPointAndNormalOnCube(geoms[iter],time*index, lightSourceSample, normal);
				}

				// Diffuse Lighting Calculation
				glm::vec3 L = glm::normalize(lightSourceSample - minIntersectionPoint);
				
				//Shadow Ray check
				ray shadowRay;
				shadowRay.origin = minIntersectionPoint + NUDGE * L;
				shadowRay.direction = L;

				bool visible = visibilityCheck(shadowRay,geoms,numberOfGeoms,lightSourceSample, intersectedGeom, iter);

				if(visible)
				{
					diffuseLight += lightMaterial.color * lightMaterial.emittance * glm::max(glm::dot(L,minNormal),0.0f);

					
					// Calculate Phong Specular Part only if exponent is greater than 0
					if(m.specularExponent > FLOAT_EPSILON)
					{
						glm::vec3 reflectedLight = 2.0f * minNormal * glm::dot(minNormal, L) - L;
						phongLight += lightMaterial.color * lightMaterial.emittance * pow(glm::max(glm::dot(reflectedLight,minNormal),0.0f),m.specularExponent);
					}

				}
			}
		}
#endif;

		AbsorptionAndScatteringProperties absScatProps;
		glm::vec3 colorSend, unabsorbedColor;
		ray returnRay = r;
		int rayPropogation = calculateBSDF(returnRay,minIntersectionPoint,minNormal,diffuseColor*m.emittance,absScatProps,colorSend,unabsorbedColor,m);
		
		// Diffuse Reflection or light source
		if (rayPropogation == 0)
		{
#if PATHTRACER
#if COMPACTION
			colors[r.pixelIndex] += r.transmission * emittance;
#else
			colors[index] += r.transmission * emittance;
#endif
#if PAINTERLY
			glm::vec3 randomVector = generateRandomNumberFromThread(resolution,time * (rayDepth+1),1,1);
#else
			glm::vec3 randomVector = generateRandomNumberFromThread(resolution,time * (rayDepth+1),x,y);
#endif
			r.direction = calculateRandomDirectionInHemisphere(minNormal, randomVector.x, randomVector.y);
			r.origin = minIntersectionPoint + 0.0005f * r.direction;
			r.transmission *= diffuseColor;
			
#else
#if COMPACTION
			colors[r.pixelIndex] += r.transmission * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
#else
			colors[index] += r.transmission * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
#endif
			r.transmission = glm::vec3(0);
#endif
			rayPool[index] = r;

		}
		// Reflection; calculate transmission coeffiecient
		else if(rayPropogation == 1)
		{
#if PATHTRACER
#if COMPACTION
			colors[r.pixelIndex] += r.transmission * emittance;
#else
			colors[index] += r.transmission * emittance;
#endif
#else
#if COMPACTION
			colors[r.pixelIndex] += r.transmission * (1.0f - m.hasReflective) * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
#else
			colors[index] += r.transmission * (1.0f - m.hasReflective) * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
#endif
#endif
			returnRay.transmission = r.transmission * diffuseColor *  m.hasReflective;
			rayPool[index] = returnRay;
		}
		// Refraction; calculate transmission coeffiecient
		else if (rayPropogation == 2)
		{

#if PATHTRACER
#if COMPACTION
			colors[r.pixelIndex] += r.transmission * emittance;
#else
			colors[index] += r.transmission * emittance;
#endif
#else
#if COMPACTION
			colors[r.pixelIndex] += r.transmission * (1.0f - m.hasRefractive) * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
#else
			colors[index] += r.transmission * (1.0f - m.hasRefractive) * ( emittance + diffuseLight * diffuseColor +  phongLight * specularColor);
#endif
#endif
			returnRay.transmission = r.transmission * diffuseColor * m.hasRefractive;


#if FRESNEL
			// Fresnel Calculation

			// Fabs because the angle is always between 0 and 90, direction not-withstanding
			float nd = fabs(glm::dot(r.direction, minNormal));
			float nt = fabs(glm::dot(returnRay.direction, minNormal));
			float n_a = nd < 0 ? 1.0f : m.indexOfRefraction;
			float n_b = nd < 0 ? m.indexOfRefraction : 1.0f;
			float amountReflected;

#if SCHLICK
			// Schlick's Approximation

			float RO = (n_a - n_b) * (n_a - n_b) / ( (n_a + n_b) * (n_a + n_b));
			float c;
			if(n_a < n_b)
				c = 1 - nd;
			else
				c = 1 - nt;

			amountReflected = RO + (1-RO) * c * c * c * c * c;

#else
			// Fresnels equations
			float reflectedParallel = (n_b * nd - n_a * nt) * (n_b * nd - n_a * nt) / ((n_b * nd + n_a * nt) * (n_b * nd + n_a * nt));
			float reflectedPerpendicular = (n_a * nd - n_b * nt) * (n_a * nd - n_b * nt) / ((n_a * nd + n_b * nt) * (n_a * nd + n_b * nt));
			amountReflected = 0.5 * (reflectedParallel + reflectedPerpendicular);
#endif
			// Stochastically decide whether to reflect or refract
			glm::vec3 randVector = generateRandomNumberFromThread(resolution,time * (rayDepth+1),x,y);
			
			// If a uniform variable is less than the reflected amount, this ray shall be reflected
			if(randVector.y  < amountReflected)
			{
				returnRay.direction = r.direction - 2.0f * minNormal  * glm::dot(minNormal,r.direction);
				returnRay.origin = minIntersectionPoint + NUDGE * returnRay.direction;
			}
#endif
			rayPool[index] = returnRay;
		}
	}
	// No intersection, mark rays as dead
	// Ambeint term 
	else
	{
		glm::vec3 ambient = glm::vec3(0,0,0);
#if COMPACTION
		colors[r.pixelIndex] += ambient;
#else
		colors[index] += ambient; 
#endif
		r.transmission = glm::vec3(0.0f);
		rayPool[index] = r;
	}
	
	/*
		//Checking for correct ray direction
		colors[index].x = fabs(r.direction.x);
		colors[index].y = fabs(r.direction.y);
		colors[index].z = fabs(r.direction.z);
	
		//Check for correct material pickup
		colors[index] = color;

		//Checking for correct depth testing
		colors[index] = color * (maxDepth - minDepth)/maxDepth;

		//Checking for correct normals
		colors[index] = glm::vec3(minNormal);
		colors[index] = glm::vec3( fabs(minNormal.x), fabs(minNormal.y), fabs(minNormal.z));
	*/
	
   }
}

__global__ void fillRayPoolFromCamera(glm::vec2 resolution, float time, cameraData cam, ray* rayPool){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){

    
	ray r;
	r = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov
#if DOF
											,cam.aperture, cam.focusPlane
#endif
		);

#if COMPACTION
	r.pixelIndex = index;
#endif
	r.transmission = glm::vec3(1.0f);

	// Access global memory only once
	rayPool[index] = r;
   }
}

__global__ void combineIntoAccumulatorImage(glm::vec2 resolution, float frames, glm::vec3* inputColors, glm::vec3* displayColors)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
	  //displayColors[index] = (((frames-1) * displayColors[index]) + inputColors[index])/frames;

	  // Averaging done in SendToPBO
	  displayColors[index] += inputColors[index];
  }
}


#if COMPACTION

/***

	Compaction:
	- Mark whether ray should be compacted or not based on transmission
	- Inclusive Scan over predicate array (in place?) to generate output indices
	  Now all rays corresponding to an ouput index > 0 should go to index outputIndex - 1 in output array
	- Use a scatter operation to distribute ray data onto those locations in the output array.

***/

// NVidiaScan
__global__ void prescan(float *g_odata, float *g_idata, int n) 
{

	int bid = blockIdx.x;

	// must give final sum into auxilary array at bid.
	// maximum number of elements that can be prescanned?
	// number of threads per block * number of threads per block (each scan per block and then one block to scan all the in between things)
	// @TODO: how to scan more than the above described limit?

	// NVIDIA implenentation follows from: 
	// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

	extern __shared__ float temp[];// allocated on invocation 
	int thid = threadIdx.x; 
	int offset = 1; 
	temp[2*thid] = g_idata[2*thid]; // load input into shared memory 
	temp[2*thid+1] = g_idata[2*thid+1]; 
	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree 
	{ 
		__syncthreads(); 
		if (thid < d) 
		{ 
			int ai = offset*(2*thid+1)-1; 
			int bi = offset*(2*thid+2)-1; 
			temp[bi] += temp[ai]; 
		} 
		offset *= 2; 
	} 
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element 
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
	{ 
		offset >>= 1; 
		__syncthreads(); 
		if (thid < d) 
		{ 
			int ai = offset*(2*thid+1)-1; 
			int bi = offset*(2*thid+2)-1; 
			float t = temp[ai]; 
			temp[ai] = temp[bi]; 
			temp[bi] += t; 
		} 
	} 
	__syncthreads(); 
	g_odata[2*thid] = temp[2*thid]; // write results to device memory 
	g_odata[2*thid+1] = temp[2*thid+1]; 
}

// Mark with predicate whether active or inactive
__global__ void predicateMark(ray* inputRays, int* outputPredicate, int size)
{
	// Using 1D kernel for compaction
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < size)
	{
		if(glm::length(inputRays[index].transmission) > FLOAT_EPSILON)
		{
			outputPredicate[index] = 1;
		}
		else
		{
			outputPredicate[index] = 0;
		}
	}
}

// Scan Per Block
__global__ void naiveScanPerBlock(int *inData, int* outData, int *blockSum, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int threadId = threadIdx.x;

	// third arg of kernel launch = numFloats*sizeofFloat
	extern __shared__ int sharedElements[];

	// Fill elements into shared Memory
	if(index < size)
		sharedElements[threadId] = inData[index];
	// If last block, fill in zeros for padding
	else
		sharedElements[threadId] = 0;

	__syncthreads();

	int numSteps = ceil(log(1.0f*blockDim.x)/log(2.0f));
	int offset = 1;
	for(int d = 0; d < numSteps; d++, offset *= 2)
	{
		int t = sharedElements[threadId];
		if(threadId >= offset)
		{
			t = t + sharedElements[threadId - offset];
		}
		__syncthreads();
		
		sharedElements[threadId] = t;
		__syncthreads();
	}
	
	outData[index] = sharedElements[threadId];
	if(threadIdx.x == blockDim.x-1)
	{
		blockSum[blockIdx.x] = sharedElements[threadId];
	}
}

// Add back Scanned blocksums
__global__ void addBackBlockSums(int *outData, int *blockSum, int size, int *returnedSum)
{
	int index = blockIdx.x * blockDim.x  +threadIdx.x;
	if(blockIdx.x > 0)
		outData[index] = outData[index] + blockSum[blockIdx.x-1];
	if(index == size - 1)
		returnedSum[0] = outData[index];
}

// Wrap around total scanned array
int parallelScan(int *inData, int *outData, int size, int d=0)
{
	if(size==0)
		return 0;

	int threads = 1024;
	int blocks = ceil(size*1.0f/threads);
	dim3 blocksPerGrid(blocks,1,1);
	dim3 threadsPerBlock(threads,1,1);

	//std::cout<<"ParallelScan in "<<blocks<<" blocks of "<<threads<<" threads\n";

	int *cudaBlockSum;
	cudaMalloc((void**)&cudaBlockSum, blocks*sizeof(int));

	int *cudaBlockSumScan;
	cudaMalloc((void**)&cudaBlockSumScan, blocks*sizeof(int));

	int *sum;
	sum = (int*)malloc(sizeof(int));
	sum[0] = 0;
	int *cudaSum;
	cudaMalloc((void**)&cudaSum,sizeof(int));

	naiveScanPerBlock<<<blocksPerGrid, threadsPerBlock, threads*sizeof(int)>>>(inData, outData,cudaBlockSum,size);
	checkCUDAError("Naive Scan Failed!");	

	// If the number of blocks is 1
	//if()
	//	only do naive scan per block for one block and copy into cudaBlockSumScan
	//	naiveScanPerBlock<<<1,threadsPerBlock, threads*sizeof(float)>>>(....)
	//else			recursive
	//	parallelScan(cudaBlockSum,cudaBlockSumScan,blocks);

	// Base Case of recursion
	if (blocks==1)
	{
		cudaMemcpy(cudaSum,cudaBlockSum,sizeof(int),cudaMemcpyDeviceToDevice);
		//return 0;
	}
	else
	{
		parallelScan(cudaBlockSum,cudaBlockSumScan,blocks,d+1);
		addBackBlockSums<<<blocksPerGrid, threadsPerBlock>>>(outData, cudaBlockSumScan, size, cudaSum);
		checkCUDAError("Add Back Blocks Failed!");	
	}
	

	if(d==0)
	{
		cudaMemcpy(sum,cudaSum,sizeof(int),cudaMemcpyDeviceToHost);
	}

	cudaFree(cudaSum);
	cudaFree(cudaBlockSumScan);
	cudaFree(cudaBlockSum);
	return (*sum);

}

// Scatter Rays into the appropriate locations in the output array
__global__ void scatter(ray* inputRays, ray* outputRays, int* predicate, int* scatterIndices, int size)
{
	// Using 1D kernel for compaction
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(index < size)
	{
		int scatterIndex = scatterIndices[index];
		// Data with scatter index < 0 is data not to be used
		if(predicate[index] > 0)
		{
			outputRays[scatterIndex - 1] = inputRays[index];
		}
	}
}

// Stream Compaction
int streamCompactRayPool(ray* inputRays, ray* outputRays, int size)
{

	/*
	// Testing recursive scan primitive

	int inputData[] = {1,1,1,1,1,1,1,1};//{1,1,2,3,4,5,6,7};
	int dataSize = 8;

	int *cudaInputData;
	cudaMalloc((void**)&cudaInputData,dataSize*sizeof(int));
	cudaMemcpy(cudaInputData,inputData,dataSize*sizeof(int),cudaMemcpyHostToDevice);

	int *cudaOutputData;
	cudaMalloc((void**)&cudaOutputData,dataSize*sizeof(int));

	int val = parallelScan(cudaInputData,cudaOutputData,dataSize);


	std::cout<<"Number of elements: "<<val<<std::endl;

	int *outputData;
	outputData = (int*)malloc(dataSize*sizeof(int));
	cudaMemcpy(outputData,cudaOutputData,dataSize*sizeof(int),cudaMemcpyDeviceToHost);

	for(int i=0; i<dataSize; i++)
		std::cout<<outputData[i]<<", ";

	std::cout<<std::endl;

	//free(inputData);
	free(outputData);
	cudaFree(cudaInputData);
	cudaFree(cudaOutputData);

	getchar();

	*/

	int* predicateArray;
	cudaMalloc((void**)&predicateArray,size*sizeof(int));

	int* scatterLocations;
	cudaMalloc((void**)&scatterLocations,size*sizeof(int));


	int numThreads = 512;
	int numBlocks = ceil(size*1.0f/numThreads);

	// Mark Predicates
	predicateMark<<<dim3(numBlocks,1,1),dim3(numThreads,1,1)>>>(inputRays, predicateArray, size);
	checkCUDAError("Predicate Mark Failed!");	

	// Scan Predicate to get location and also total number of final rays
	int compactedSize = parallelScan(predicateArray,scatterLocations,size);

	// Scatter rays to new locations in output array
	scatter<<<dim3(numBlocks,1,1),dim3(numThreads,1,1)>>>(inputRays,outputRays,predicateArray,scatterLocations,size);
	checkCUDAError("Scatter Failed!");	

	cudaFree(scatterLocations);
	cudaFree(predicateArray);
	
	return compactedSize;
}


#endif



#if PHOTONMAP

// Create a helper function to call these functions

//function for emitting photons from a sphere light
__global__ void emitPhotons(photon* photonPool, int numPhotons, int numBounces, staticGeom* geoms, int* lights, int numberOfLights,
														float* cudaAccumLightProbabilities, material* materials, float time)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < numPhotons)
	{
		photon p;
		
		// Do a random-check to choose a certain light: take into consideration area of lights
		// Do a better random generation
		glm::vec3 randoms = generateRandomNumberFromThread(glm::vec2(800,800),time,index,numberOfLights);
		
		// Pick light based on cudaAccumLightProbabilities
		int lightIndex;
		bool largerThanPrev = true;			//whether randoms.x is larger than the previous probability in cudaAccumLightProbabilities
		for (int i=0; i<numberOfLights; ++i) {
			if (randoms.x <= cudaAccumLightProbabilities[i]) {
				if (largerThanPrev) {
					lightIndex = i;
					break;
				}
				else {
					largerThanPrev = true;
				}
			}
		}

		staticGeom lightChosen = geoms[lights[lightIndex]];			//get the light using the index from the lights array
		
		// for now only supports sphere and cube lights
		glm::vec3 position, normal;
		if(lightChosen.type == SPHERE)
		{
			getRandomPointAndNormalOnSphere(lightChosen,index, position, normal);
		}
		else if (lightChosen.type == CUBE)
		{
			getRandomPointAndNormalOnCube(lightChosen,index, position, normal);
		}
		p.position = position;

		// Shooting direction is normal at the point or random direction?
		// I think the lecture said choose random direction.
		p.dout = calculateRandomDirectionInHemisphere(normal,randoms.y,randoms.z);
		p.din = glm::vec3(0.0f);
		
		
		// Set color of photon
		material lightMaterial = materials[lightChosen.materialid];
		p.color = lightMaterial.emittance * lightMaterial.color;

		// Set whether photon has been stored/absorbed (dead)
		p.stored = false;
		p.geomid = lights[lightIndex];
		p.bounces = 0;		//increment the number of bounces by 1

		photonPool[index] = p;

		//set the rest of the photons in the array to not stored
		for (int i = 1; i < numBounces; ++i) {
			photon placeHolder;
			placeHolder.color = glm::vec3(0.0f);
			placeHolder.stored = false;
			placeHolder.bounces = -1;
			photonPool[numPhotons * i + index] = placeHolder;
		}
	}

}

__global__ void displayPhotons(photon* photonPool, int numPhotons, int numBounces, glm::vec2 resolution, cameraData cam, glm::vec3* colors, cudaMat4 viewProjectionViewport, float flux)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index < numPhotons)
	{
		
		for (int i = 0; i < numBounces; ++i) {
		
			photon p = photonPool[numPhotons * i + index];

			//only display color if photon is not dead at that position
			if (p.stored) {
				glm::vec3 photonToEye = cam.position - p.position;
				
				// Do this assuming view, projection and viewport matrices are provided
				//glm::vec3 screenPosition = multiplyMV(viewProjectionViewport,glm::vec4(p.position,1.0f));
				
				glm::vec4 screenPosition = multiplyMV_4(viewProjectionViewport, glm::vec4(p.position, 1.0f));

				// Shift to viewport matrix
				//transform to clip
				screenPosition.x /= screenPosition.w;
				screenPosition.y /= screenPosition.w;
				screenPosition.z /= screenPosition.w;

				//transform to screen coord
				screenPosition.x = (screenPosition.x+1) * resolution.x/2.0f;
				screenPosition.y = (-screenPosition.y+1) * resolution.y/2.0f;

				if(screenPosition.x >=0 && screenPosition.x < resolution.x && screenPosition.y >=0 && screenPosition.y < resolution.y)
				{
					// write to the color buffer!
					// race conditions?
					int x = screenPosition.x;
					int y = screenPosition.y;
					int pixelIndex = x + (y * resolution.x);
					//colors[pixelIndex] = glm::abs(p.dout);		//glm::abs causes a kernel failure on my computer...
					colors[pixelIndex] = p.color;
				}
			}

		}
	}
}

__global__ void testImage(glm::vec3* colors, glm::vec2 resolution) {
	
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if (x <= resolution.x && y <= resolution.y) { 
		colors[index] = glm::vec3(1.0);
	}

}

__global__ void bouncePhotons(photon* photonPool, int numPhotons, int currentBounces, staticGeom* geoms, int numberOfGeoms, material* materials, float time)
{
	//bounce photons around
	int index = blockIdx.x * blockDim.x + threadIdx.x; 

	if (index < numPhotons){

		int prevIndex = index;
		if (currentBounces!=0)
			prevIndex = index + (currentBounces-1) * numPhotons;

		int nextIndex = index + currentBounces * numPhotons;

		//load a photon from memory
		photon p = photonPool[prevIndex];

		//create ray using photon
			ray r;
			r.origin = p.position + 0.01f*p.dout;		//offset point a little to avoid self intersection
			r.direction = p.dout;

			//intersection testing
			int intersectedGeom = -1;
			int intersectedMaterial = -1;
			float minDepth = 1000000.0f;
			glm::vec3 minIntersectionPoint;
			glm::vec3 minNormal = glm::vec3(0.0f);
			
			for (int iter=0; iter < numberOfGeoms; iter++)
			{
					float depth=-1;
					glm::vec3 intersection;
					glm::vec3 normal;
					staticGeom currentGeometry = geoms[iter];
					if (currentGeometry.type == CUBE)
					{
							depth = boxIntersectionTest(currentGeometry,r,intersection,normal);
					}
					
					else if (geoms[iter].type == SPHERE)
					{
							depth = sphereIntersectionTest(currentGeometry,r,intersection,normal);
					}
					

					if (depth > 0 && depth < minDepth)
					{
							minDepth = depth;
							minIntersectionPoint = intersection;
							minNormal = normal;
							intersectedGeom = iter;
							intersectedMaterial = currentGeometry.materialid;
					}
			}

			//if intersection occurs, accumulate color and keep bouncing
			if (intersectedGeom > -1) {

				material m = materials[intersectedMaterial];

				//assume diffuse surfaces only for now, so bounce in random direction
				p.color *= m.color;

				glm::vec3 randoms = generateRandomNumberFromThread(glm::vec2(800,800),time,index,currentBounces+1);
				p.din = p.dout;
				//p.stored = true;
				p.dout = calculateRandomDirectionInHemisphere(minNormal,randoms.y,randoms.z);
				p.position = minIntersectionPoint;

				AbsorptionAndScatteringProperties absScatProps;
				glm::vec3 colorSend, unabsorbedColor;
				ray returnRay = r;
				
				int rayPropogation = calculateBSDF(returnRay,minIntersectionPoint,minNormal,p.color,absScatProps,colorSend,unabsorbedColor,m);

				// Reflection; calculate transmission coeffiecient
				if(rayPropogation == 1)
				{
					p.dout = returnRay.direction;
					p.color = p.color * m.hasReflective;
					p.stored = true;
				}
				// Refraction; calculate transmission coeffiecient
				else if (rayPropogation == 2)
				{
					p.color = p.color * m.hasRefractive;
					p.stored = true;

#if FRESNEL
					// Fresnel Calculation

					// Fabs because the angle is always between 0 and 90, direction not-withstanding
					float nd = fabs(glm::dot(r.direction, minNormal));
					float nt = fabs(glm::dot(returnRay.direction, minNormal));
					float n_a = nd < 0 ? 1.0f : m.indexOfRefraction;
					float n_b = nd < 0 ? m.indexOfRefraction : 1.0f;
					float amountReflected;

#if SCHLICK
					// Schlick's Approximation

					float RO = (n_a - n_b) * (n_a - n_b) / ( (n_a + n_b) * (n_a + n_b));
					float c;
					if(n_a < n_b)
						c = 1 - nd;
					else
						c = 1 - nt;

					amountReflected = RO + (1-RO) * c * c * c * c * c;

#else
					// Fresnels equations
					float reflectedParallel = (n_b * nd - n_a * nt) * (n_b * nd - n_a * nt) / ((n_b * nd + n_a * nt) * (n_b * nd + n_a * nt));
					float reflectedPerpendicular = (n_a * nd - n_b * nt) * (n_a * nd - n_b * nt) / ((n_a * nd + n_b * nt) * (n_a * nd + n_b * nt));
					amountReflected = 0.5 * (reflectedParallel + reflectedPerpendicular);
#endif
					// Stochastically decide whether to reflect or refract
					glm::vec3 randVector = generateRandomNumberFromThread(glm::vec2(637,791),time,index,currentBounces+1);

					// If a uniform variable is less than the reflected amount, this ray shall be reflected
					if(randVector.y  < amountReflected)
					{
						p.dout = r.direction - 2.0f * minNormal  * glm::dot(minNormal,r.direction);
					}
					else
					{
						p.dout = returnRay.direction;
					}
#endif

				}
				// Default to diffuse
				else
				{
					p.stored = true;
				}
				
				p.geomid = intersectedGeom;
			}
			else {
				//kill the photon if it doesn't intersect with anything
				//When using stream compaction, need to figure if photon is stored or dead or (alive and kicking)
				p.stored = false;
				p.geomid = -1;
			}

			p.bounces ++;
			//write new bounced photon into memory
			
			photonPool[nextIndex] = p;
		}

}

#define oneOverSqrtTwoPi 0.3989422804f
__device__ float gaussianWeight( float dx, float radius)
{
	float sigma = radius/3.0;
	return (oneOverSqrtTwoPi / sigma) * exp( - (dx*dx) / (2.0 * sigma * sigma) );
}

// Caculate radiances from photons
__global__ void gatherPhotons(glm::vec2 resolution, float time, cameraData cam, glm::vec3* colors, staticGeom* geoms,
															int numberOfGeoms, ray* rayPool, photon* photons, int numPhotons, int numBounces, float flux) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if((x<=resolution.x && y<=resolution.y) && glm::length(rayPool[index].transmission) > FLOAT_EPSILON){
		ray r = rayPool[index];	

		//Check all geometry for intersection
		int intersectedGeom = -1;
		int intersectedMaterial = -1;
		float minDepth = 1000000.0f;
		glm::vec3 minIntersectionPoint;
		glm::vec3 minNormal = glm::vec3(0.0f);
		for(int iter=0; iter < numberOfGeoms; iter++)
		{
				float depth=-1;
				glm::vec3 intersection;
				glm::vec3 normal;
				staticGeom currentGeometry = geoms[iter];
				if(currentGeometry.type == CUBE)
				{
					depth = boxIntersectionTest(currentGeometry,r,intersection,normal);
				}
		
				else if(geoms[iter].type == SPHERE)
				{
					depth = sphereIntersectionTest(currentGeometry,r,intersection,normal);
				}
		

				if(depth > 0 && depth < minDepth)
				{
					minDepth = depth;
					minIntersectionPoint = intersection;
					minNormal = normal;
					intersectedGeom = iter;
					intersectedMaterial = currentGeometry.materialid;
				}

		}

		//Calculate radiance if any geometry is intersected
		if(intersectedGeom > -1)
		{
			glm::vec3 accumColor(0);
			//Use brute force search to find the photons that are within a certain radius
			for (int i=0; i<numPhotons * numBounces; ++i) {
				photon p = photons[i];
				float photonDistance  = glm::distance(minIntersectionPoint, p.position);
				// Indirect Illumination only?
				if ( photonDistance <= RADIUS && p.geomid == intersectedGeom && p.bounces > 0) {
					//Is lambert brdf cos(theta_i)?
					accumColor += gaussianWeight(photonDistance, RADIUS) *  max(0.0f, glm::dot(minNormal, -p.din)) * p.color;
				}
			}
			colors[index] += accumColor * flux;
		}
	}
}


void tracePhotons(int photonThreadsPerBlock, int photonBlocksPerGrid, photon* cudaPhotonPool,
									int numPhotons, staticGeom* cudaGeoms, int numberOfGeoms, material* cudaMaterials,
									float time)
{
	for(int i=0; i < numBounces; i++)
	{
		// Bounce Photons Around
		bouncePhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons, i,	cudageoms, numGeoms, cudamaterials, time);

#if COMPACTION
		// Do some compaction

#endif
	}
}


void initPhotonMap()
{
	//Create Memory for RayPool
	cudaPhotonPool = NULL;
	cudaMalloc((void**)&cudaPhotonPool, numBounces * numPhotons * sizeof(photon));
	
}

void cleanPhotonMap()
{
	cudaFree(cudaPhotonPool);
}

//allocate memory for geometry data
void cudaAllocateMemory(camera* renderCam, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms) {

	int size = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
	
	//send image to GPU
	cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	//package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	std::vector<int> lightVec;
	numGeoms = numberOfGeoms;

	//get geom from frame 0
	for(int i=0; i<numberOfGeoms; i++){
			staticGeom newStaticGeom;
			newStaticGeom.type = geoms[i].type;
			newStaticGeom.materialid = geoms[i].materialid;
			newStaticGeom.translation = geoms[i].translations[0];
			newStaticGeom.rotation = geoms[i].rotations[0];
			newStaticGeom.scale = geoms[i].scales[0];
			newStaticGeom.transform = geoms[i].transforms[0];
			newStaticGeom.inverseTransform = geoms[i].inverseTransforms[0];
			geomList[i] = newStaticGeom;

			//store which objects are lights
			if(materials[geoms[i].materialid].emittance > 0)
					lightVec.push_back(i);
	}

	cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	//copy materials to memory
	cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	//copy light ID to memeory
	numLights = lightVec.size();

	int* lightID = new int[numLights];
	for(int i = 0; i <numLights; ++i)
			lightID[i] = lightVec[i];

	cudaLights = NULL;
	cudaMalloc((void**)&cudaLights, numLights*sizeof(int));
	cudaMemcpy( cudaLights, lightID, numLights*sizeof(int), cudaMemcpyHostToDevice);

#if PHOTONMAP
	//std::cout<<"allocating mem for photon pool"<<std::endl;
	cudaPhotonPool = NULL;
	cudaMalloc((void**)&cudaPhotonPool, numBounces * numPhotons * sizeof(photon));

	cudaPhotonMapImage = NULL;
	cudaMalloc((void**)&cudaPhotonMapImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

	// compute the accumulated probablity of photons being emitted from each light
	float totalEmittanceTimesArea = 0;
	float* accumAccumLightProbabilities = new float[numLights];
	for (int i=0; i<numLights; ++i) {
		staticGeom light = geomList[lightID[i]];
		material lightmtl = materials[light.materialid];

		float area;
		if (light.type == SPHERE) {
			// compute the surface area of an ellipsoid
			float a = 0.5 * light.scale.x;
			float b = 0.5 * light.scale.y;
			float c = 0.5 * light.scale.z;
			float p = 1.6075;
			float ap = pow(a, p);
			float bp = pow(b, p);
			float cp = pow(c, p);
			area = 4 * PI_F * pow((ap*bp + ap*cp + bp*cp)/3, 1/p); //the approximate formula for surface area of ellipsoids
		}
		else if (light.type == CUBE) {
			// compute the surface area of a box
			float sx = light.scale.x;
			float sy = light.scale.y;
			float sz = light.scale.z;
			area = 2 * sx * sy + 2 * sx * sz + 2 * sy * sz;
		}
		totalEmittanceTimesArea += area * lightmtl.emittance;
		accumAccumLightProbabilities[i] = totalEmittanceTimesArea;
	}

	// divide accumulated "emittance * area" of each light by totalEmittanceTimesArea to get the probablity of being emitted from each light
	for (int i=0; i<numLights; ++i) {
		accumAccumLightProbabilities[i] /= totalEmittanceTimesArea;
	}

	cudaAccumLightProbabilities = NULL;
	cudaMalloc((void**)&cudaAccumLightProbabilities, numLights*sizeof(float));
	cudaMemcpy( cudaAccumLightProbabilities, accumAccumLightProbabilities, numLights*sizeof(float), cudaMemcpyHostToDevice);
#endif

	cudaAllocateAccumulatorImage(renderCam);

	delete[] geomList;
	delete[] lightID;
}

//free up memory
void cudaFreeMemory() {

	cudaFree( cudaimage);
	cudaFree( cudageoms );
	cudaFree( cudamaterials);
	cudaFree( cudaLights);

#if PHOTONMAP
	cudaFree(cudaPhotonPool);
	cudaFree(cudaPhotonMapImage);
#endif

	cudaFreeAccumulatorImage();

}


void cudaPhotonMapCore(camera* renderCam, int frame, int iterations, uchar4* PBOpos, cameraData liveCamera)
{

	// Set up crucial magic
	glm::vec2 resolution = renderCam->resolution;
	int tileSize = 8;
	dim3 pixelThreadsPerBlock(tileSize, tileSize);
	dim3 pixelBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

	//package camera data
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;
	cam.focusPlane = renderCam->focusPlanes[frame];
	cam.aperture = renderCam->apertures[frame];

	//user interaction
	cam.position +=  (liveCamera.position);
	cam.view = glm::normalize(cam.view + liveCamera.view);
	cam.aperture += liveCamera.aperture;
	cam.focusPlane += liveCamera.focusPlane;

	// Clear photon image buffer
	clearImage<<<pixelBlocksPerGrid,pixelThreadsPerBlock>>>(resolution, cudaPhotonMapImage);
	cudaThreadSynchronize();
	checkCUDAError("clearImage kernel failed!");

	// Generate Photon List
	int photonThreadsPerBlock = 512;
	int photonBlocksPerGrid = ceil(numPhotons * 1.0f/photonThreadsPerBlock);

	emitPhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons, numBounces, cudageoms, 
		cudaLights, numLights, cudaAccumLightProbabilities, cudamaterials, iterations);
	cudaThreadSynchronize();
	checkCUDAError("emit photons kernel failed!");

	// Trace all photons with all bounces
	tracePhotons(photonThreadsPerBlock, photonBlocksPerGrid, cudaPhotonPool, numPhotons, cudageoms, numGeoms, cudamaterials, iterations);
	cudaThreadSynchronize();
	checkCUDAError("tracePhotons kernel failed!");

	// Assume each light emits the same number of photons, calculate the flux per photon
	float flux = totalEnergy/((float)numPhotons/(float)numLights);


if (mode == DISP_GATHER)
{
	// Compute radiance from photons
	// Generate rays first
	ray* cudarays = NULL;
	cudaMalloc((void**)&cudarays, (renderCam->resolution.x * renderCam->resolution.y) * sizeof(ray));
	fillRayPoolFromCamera<<<pixelBlocksPerGrid, pixelThreadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudarays);
	cudaThreadSynchronize();
	checkCUDAError("fill ray pool kernel failed!");

	// Assume each light emits the same number of photons, calculate the flux per photon
	float flux = totalEnergy/(float)numPhotons;

	gatherPhotons<<<pixelBlocksPerGrid, pixelThreadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudaPhotonMapImage, cudageoms, numGeoms,
		cudarays, cudaPhotonPool, numPhotons, numBounces, flux);
	cudaThreadSynchronize();
	checkCUDAError("gather photonskernel failed!");

	cudaFree(cudarays);
	cudaThreadSynchronize();
	checkCUDAError("free ray pool failed!");
}
else if (mode == DISP_PHOTONS)
{
	// Calculate Viewport * Projection * View matrix from camera info
	glm::vec3 center = cam.position + cam.view;

	glm::mat4 viewMat = glm::lookAt(cam.position, center, cam.up);
	glm::mat4 projectionMat = glm::perspective(cam.fov.y*2, cam.resolution.x/cam.resolution.y, 0.1f, 1000.0f);

	cudaMat4 viewProjectionViewPort = utilityCore::glmMat4ToCudaMat4(projectionMat*viewMat);
	
	//utilityCore::printCudaMat4(viewProjectionViewPort);

	// Display all photons in the photonImage buffer
	displayPhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons, numBounces, resolution, 
		cam, cudaPhotonMapImage, viewProjectionViewPort, flux);
	cudaThreadSynchronize();
	checkCUDAError("display photons kernel failed!");
}


#if ACCUMULATION
	combineIntoAccumulatorImage<<<pixelBlocksPerGrid,pixelThreadsPerBlock>>>(renderCam->resolution, (float)iterations, cudaPhotonMapImage, accumulatorImage);
	sendImageToPBO<<<pixelBlocksPerGrid,pixelThreadsPerBlock>>>(PBOpos, renderCam->resolution, accumulatorImage, (float)iterations);
#else
	sendImageToPBO<<<pixelBlocksPerGrid,pixelThreadsPerBlock>>>(PBOpos, renderCam->resolution, cudaPhotonMapImage, 1.0f);
#endif
	cudaThreadSynchronize();
	checkCUDAError("Send to PBO kernel failed!");

	//retrive image from GPU
	int imageSize = (int)resolution.x * (int) resolution.y;
#if ACCUMULATION
	cudaMemcpy(renderCam->image, accumulatorImage, imageSize*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
	cudaMemcpy(renderCam->image, cudaPhotonMapImage, imageSize*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif

	cudaThreadSynchronize();
	checkCUDAError("Photon mapping kernel failed!");

}
#endif	

void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, cameraData liveCamera){

  // testing 
  //streamCompact();

  //if(iterations == 0)
  //{
  //  // Allocate Accumulator Image
  //  cudaAllocateAccumulatorImage(renderCam);
  //}

  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.aperture = renderCam->apertures[frame];
  cam.focusPlane = renderCam->focusPlanes[frame];
  cam.fov = renderCam->fov;

  //user interaction
  cam.position +=  (liveCamera.position);
  cam.view = glm::normalize(cam.view + liveCamera.view);
  cam.aperture += liveCamera.aperture;
  cam.focusPlane += liveCamera.focusPlane;

  //Create Memory for RayPool
  ray* cudarays = NULL;
  cudaMalloc((void**)&cudarays, (renderCam->resolution.x * renderCam->resolution.y) * sizeof(ray));

#if COMPACTION
  ray* cudarays2 = NULL;
  cudaMalloc((void**)&cudarays2, (renderCam->resolution.x * renderCam->resolution.y) * sizeof(ray));
#endif

  //clear On screen buffer
  clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution, cudaimage);

  //Fill ray pool with rays from camera for first iteration
  fillRayPoolFromCamera<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudarays);
  int numberOfRays = (int)renderCam->resolution.x * (int)renderCam->resolution.y;

  //std::cout<<"StreamCompaction: ";

  int linearTileSize = tileSize*tileSize;
  for(int i=0; i < MAX_RECURSION_DEPTH && numberOfRays > 0; i++)
  {
#if COMPACTION
	    dim3 linearGridSize((int)ceil(numberOfRays*1.0f/linearTileSize),1,1);
		raytraceRay<<<linearGridSize, dim3(linearTileSize,1,1)>>>(renderCam->resolution, (float)iterations, cam, traceDepth+i,
															cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials,
															i%2==0?cudarays : cudarays2,
															numberOfRays);
		checkCUDAError("Ray Trace Failed!");	 

 		numberOfRays = streamCompactRayPool( i%2==0? cudarays : cudarays2,
			  						         i%2==0? cudarays2 : cudarays,
										     numberOfRays);
#else
	//kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth+i, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, cudarays);
#endif
  }

  //getchar();

#if ACCUMULATION
  combineIntoAccumulatorImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudaimage, accumulatorImage);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, accumulatorImage, (float)iterations);
#else
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, 1.0f);
#endif

  //retrieve image from GPU for sending to bmp file
  if(singleFrameMode)
#if ACCUMULATION
	cudaMemcpy( renderCam->image, accumulatorImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudarays );
#if COMPACTION
  cudaFree( cudarays2);
#endif
  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}


//Clear AccumulatorImage. For an interactive application, this needs to be called everytime the camera moves or the scene changes
void cudaClearAccumulatorImage(camera *renderCam)
{
	// set up crucial magic
  	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution, accumulatorImage);
}

//Allocate Memory For AccumulatorImage
void cudaAllocateAccumulatorImage(camera *renderCam)
{
	cudaMalloc((void**)&accumulatorImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaClearAccumulatorImage(renderCam);
}

//Free memory of the accumulator image
void cudaFreeAccumulatorImage()
{
	cudaFree(accumulatorImage);
}