// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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
#define PAINTERLY 0
#define PHOTONCOMPACT 1
#define K 10

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

int numPhotons = 10000;
int numPhotonsCompact = numPhotons;

int numBounces = 10;			//hard limit of n bounces for now
float emitEnergyScale = 1000; //Empirically Verify this value

float totalEnergy;	//total amount of energy in the scene, used for calculating flux per photon
float flux;

photon* cudaPhotonPool;		//global variable of photons

#if PHOTONCOMPACT
photon* cudaPhotonPoolCompact;		//stores output photons after stream compaction
#endif

int* cudaPhotonGridIndex;			//maps photonID to gridID
int* cudaGridFirstPhotonIndex;

#define RADIUS 1.0f

//gridAttributes grid(-5.5, -0.5, -5.5, 5.5, 10.5, 5.5, RADIUS);
gridAttributes grid(0, 0, 0, 0, 0, 0, RADIUS);		//for testing grid bounding box

glm::vec3* accumulatorImage = NULL;
extern bool singleFrameMode;
extern int mode;

//scene data
glm::vec3* cudaimage;
staticGeom* cudageoms;
material* cudamaterials;
glm::vec3* cudavertices;
glm::vec3* cudanormals;
glm::vec2* cudauvs;
triangle* cudafaces;
int* cudaLights;
float* cudaAccumLightProbabilities;
int numLights;
int numGeoms;
int numFaces;
int numVertices;
int numNormals;
int numUVs;
cudatexture* cudatextures; //textures' width, height and x index
float4* cudatexturedata; //data we bind tex to
texture<float4, 2, cudaReadModeElementType> tex; //textures of all objects in the scene

// per frame variables
ray* cudarays = NULL;
#if COMPACTION
ray* cudarays2 = NULL;
#endif

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

__device__ glm::vec3 getMaterialColor(material mtl, cudatexture* textures, glm::vec2 uv) {
	if (mtl.textureid == -1) {
		return mtl.color;
	}
	else {
		cudatexture texture = textures[mtl.textureid];
		float x = texture.xindex + uv.x * texture.width;
		float y = uv.y * texture.height;
		float4 colorData = tex2D(tex, x, y);
		return glm::vec3(colorData.x, colorData.y, colorData.z);
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

//perdicate marking for photons
__global__ void predicateMarkPhotons(photon* inputPhotons, int* outputPredicate, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		
		if (inputPhotons[index].stored) {
			outputPredicate[index] = 1;
		}
		else {
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

//scatter photons to appropriate locations
__global__ void scatterPhotons(photon* inputPhotons, photon* outputPhotons, int* predicate, int* scatterIndices, int size) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < size) {
		int scatterIndex = scatterIndices[index];
		
		// Data with scatter index < 0 is data not to be used
		if (predicate[index] > 0) {		
			outputPhotons[scatterIndex - 1] = inputPhotons[index];
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
	checkCUDAError("Ray Predicate Mark Failed!");	

	// Scan Predicate to get location and also total number of final rays
	int compactedSize = parallelScan(predicateArray,scatterLocations,size);

	// Scatter rays to new locations in output array
	scatter<<<dim3(numBlocks,1,1),dim3(numThreads,1,1)>>>(inputRays,outputRays,predicateArray,scatterLocations,size);
	checkCUDAError("Scatter Failed!");	

	cudaFree(scatterLocations);
	cudaFree(predicateArray);
	
	return compactedSize;
}


//stream compaction for photons, maybe use template later to make code cleaner
int streamCompactPhotons (photon* inputPhotons, photon* outputPhotons, int size) {

	int* predicateArray;
	cudaMalloc((void**)&predicateArray,size*sizeof(int));

	int* scatterLocations;
	cudaMalloc((void**)&scatterLocations,size*sizeof(int));

	int numThreads = 512;
	int numBlocks = ceil(size*1.0f/numThreads);

	// Mark Predicates
	predicateMarkPhotons<<<dim3(numBlocks,1,1),dim3(numThreads,1,1)>>>(inputPhotons, predicateArray, size);
	checkCUDAError("Photon Predicate Mark Failed!");	

	// Scan Predicate to get location and also total number of final rays
	int compactedSize = parallelScan(predicateArray,scatterLocations,size);

	// Scatter rays to new locations in output array
	scatterPhotons<<<dim3(numBlocks,1,1),dim3(numThreads,1,1)>>>(inputPhotons,outputPhotons,predicateArray,scatterLocations,size);
	checkCUDAError("Scatter Failed!");	

	cudaFree(scatterLocations);
	cudaFree(predicateArray);
	
	return compactedSize;

}

#endif


//gets gridcell index based on photon's position
__device__ void getCellIndex(glm::vec3 position, gridAttributes& grid, int& i, int& j, int& k) {
        i = floor((position.x - grid.xmin)/grid.cellsize);
        j = floor((position.y - grid.ymin)/grid.cellsize);
        k = floor((position.z - grid.zmin)/grid.cellsize);
}

__device__ int gridPhotonHash (int &i, int& j, int &k, gridAttributes& grid) {

	return i + j * grid.xdim + k * grid.xdim * grid.ydim;
}

//maps photon's position to a grid cell index
__global__ void mapPhotonToGrid(photon* photonPool, int numPhotons, int* gridIndex, gridAttributes grid) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numPhotons) {
		photon p = photonPool[index];

		if(p.stored == false) return;

		//find which gridcell this photon is in
		int x, y, z;
		getCellIndex(p.position, grid, x, y, z);

		if (x<0 || x>=grid.xdim || y<0 || y>=grid.ydim || z<0 || z>=grid.zdim) return;

		//use hash funcion x + y*dx + z*dx*dy to assign a grid index to photon
		gridIndex [index] = gridPhotonHash(x, y, z, grid);
	}
}

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

		photonPool[index] = p;

		//set the rest of the photons in the array to not stored
		for (int i = 1; i < numBounces; ++i) {
			photon placeHolder;
			placeHolder.color = glm::vec3(0.0f);
			placeHolder.geomid = -1;
			placeHolder.stored = false;
			photonPool[numPhotons * i + index] = placeHolder;
		}
	}

}

//for world to screen space transformations
__device__ glm::vec2 worldToScreen(cudaMat4& transMat, glm::vec4& position, glm::vec2& resolution) {
	
	glm::vec4 screenPosition = multiplyMV_4(transMat,position);

	// Shift to viewport matrix
	//transform to clip
	screenPosition.x /= screenPosition.w;
	screenPosition.y /= screenPosition.w;
	screenPosition.z /= screenPosition.w;

	//transform to screen coord
	screenPosition.x = (screenPosition.x+1) * resolution.x/2.0f;
	screenPosition.y = (-screenPosition.y+1) * resolution.y/2.0f;

	return glm::vec2 (screenPosition.x, screenPosition.y);
}

__global__ void displayPhotons(photon* photonPool, int numTotalPhotons, glm::vec2 resolution, cameraData cam, glm::vec3* colors, cudaMat4 viewProjectionViewport, float flux)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < numTotalPhotons)
	{

		photon p = photonPool[index];
		glm::vec3 photonToEye = cam.position - p.position;

		//only display color if photon is not dead at that position
		if (p.stored) {
			glm::vec3 photonToEye = cam.position - p.position;

			//transform position from world to screen space
			glm::vec2 screenPosition = worldToScreen(viewProjectionViewport, glm::vec4(p.position, 1.0f), resolution);

			if(screenPosition.x >=0 && screenPosition.x < resolution.x && screenPosition.y >=0 && screenPosition.y < resolution.y)
			{
				// write to the color buffer!
				// race conditions?
				int x = screenPosition.x;
				int y = screenPosition.y;
				int pixelIndex = x + (y * resolution.x);
				colors[pixelIndex] = p.color;
			}
		}
	}
}

__global__ void bouncePhotons(photon* photonPool, int numPhotons, int currentBounces, staticGeom* geoms, int numberOfGeoms, triangle* cudafaces,
															int numFaces, glm::vec3* cudavertices, glm::vec3* cudanormals, glm::vec2* cudauvs, material* materials,
															cudatexture* cudatextures, float time)
{
	//bounce photons around
	int index = blockIdx.x * blockDim.x + threadIdx.x; 

	if (index < numPhotons){
		//overwrite photons in the first two bounces (don't store direct illumination in the photon map)
		int prevIndex = index;
		if (currentBounces > 1)
			prevIndex = index + (currentBounces-2) * numPhotons;

		int nextIndex = index;
		if (currentBounces > 1)
			nextIndex = index + (currentBounces-1) * numPhotons;

		//load a photon from memory
		photon p = photonPool[prevIndex];

		//only bounce photon if photon is not in the void
		if (p.geomid != -1) {
			//create ray using photon
			ray r;
			r.origin = p.position + 0.01f*p.dout;		//offset point a little to avoid self intersection
			r.direction = p.dout;

			//intersection testing
			int intersectedGeom = -1;
			int intersectedMaterial = -1;
			glm::vec3 minIntersectionPoint;
			glm::vec3 minNormal = glm::vec3(0.0f);
			glm::vec2 minUV = glm::vec2(0.0f);

			getClosestIntersection(r, geoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
				minIntersectionPoint, minNormal, intersectedGeom, intersectedMaterial,minUV);

			p.geomid = intersectedGeom;
			//if intersection occurs, accumulate color and keep bouncing
			if (intersectedGeom > -1) {
				p.position = minIntersectionPoint;

				if (currentBounces <= 1) {
					p.stored = false;
				} else {
					p.stored = true;
				}

				material m = materials[intersectedMaterial];
				p.color *= getMaterialColor(m, cudatextures, minUV);

				//assume diffuse surfaces only for now, so bounce in random direction
				glm::vec3 randoms = generateRandomNumberFromThread(glm::vec2(800,800),time,index,currentBounces+3);
				p.din = p.dout;
				p.dout = calculateRandomDirectionInHemisphere(minNormal,randoms.y,randoms.z);

				AbsorptionAndScatteringProperties absScatProps;
				glm::vec3 colorSend, unabsorbedColor;
				ray returnRay = r;

				int rayPropogation = calculateBSDF(returnRay,minIntersectionPoint,minNormal,p.color,absScatProps,colorSend,unabsorbedColor,m,
													glm::vec2(800,800), time, currentBounces, threadIdx.x, blockIdx.x);

				// Reflection; calculate transmission coeffiecient
				if(rayPropogation == 1)
				{
					if (randoms.x < m.hasReflective) { // using randoms.x since we haven't used it before
						p.dout = returnRay.direction;
						p.stored = false;
					}
				}
				// Refraction; calculate transmission coeffiecient
				else if (rayPropogation == 2)
				{
					if (randoms.x < m.hasRefractive) {
						p.dout =  returnRay.direction;
						p.stored = false;
					}
				}
			}
			else {
				p.stored = false;
			}

			//write new bounced photon into memory
			photonPool[nextIndex] = p;
		}
	}
}

__global__ void initGridFirstPhotonIndex(int* sortedPhotonGridIndex, gridAttributes grid) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int z = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (x < grid.xdim && y < grid.ydim && z < grid.zdim) {
		int index = gridPhotonHash(x, y, z, grid);
		sortedPhotonGridIndex[index] = -1;
	}
}

__global__ void computeGridFirstPhoton(int* sortedPhotonGridIndex, int numPhotons, int* gridFirstPhotonIndex)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; 
	if (index < numPhotons)
	{

		// First one always has to be stored
		if (index == 0)
		{
			gridFirstPhotonIndex[sortedPhotonGridIndex[index]] = index;
		}
		else
		{
			int currGrid = sortedPhotonGridIndex[index];
			int prevGrid = sortedPhotonGridIndex[index-1];

			if(currGrid != prevGrid)
			{
				gridFirstPhotonIndex[currGrid] = index;
			}
		}
	}
}

void buildSpatialHash()
{
	// calculate hash value for each photon.

	//allocate memory for grid data, do it per iteration since size might change if we use stream compaction for photons
	cudaPhotonGridIndex = NULL;

	//break compacted photons into chunks like the uncompacted ones
	int photonThreadsPerBlock = 512;

	// stream compact photons

#if PHOTONCOMPACT 
	//stream compact photons after all bounces are finished
	numPhotonsCompact = streamCompactPhotons(cudaPhotonPool, cudaPhotonPoolCompact, numPhotons * numBounces);

	cudaMalloc((void**)&cudaPhotonGridIndex, numPhotonsCompact * sizeof(int));

	//update number of blocks needed for kernels
	int photonBlocksPerGrid = ceil(numPhotonsCompact * 1.0f / photonThreadsPerBlock);

	//map photon id to grid id
	mapPhotonToGrid<<<dim3(photonBlocksPerGrid), dim3(photonThreadsPerBlock)>>>(cudaPhotonPoolCompact, numPhotonsCompact, cudaPhotonGridIndex, grid);

#else

	//update number of blocks needed for kernels
	int photonBlocksPerGrid = ceil(numPhotons * numBounces * 1.0f / photonThreadsPerBlock);

	cudaMalloc((void**)&cudaPhotonGridIndex, numPhotons * numBounces * sizeof(int));
	mapPhotonToGrid<<<dim3(photonBlocksPerGrid), dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons * numBounces, cudaPhotonGridIndex, grid);

#endif 

	// sort photons based on hash ID
	thrust::device_ptr<int> thrustGridIndex = thrust::device_pointer_cast(cudaPhotonGridIndex);
#if PHOTONCOMPACT
	thrust::device_ptr<photon> thrustPhotons = thrust::device_pointer_cast(cudaPhotonPoolCompact);
	thrust::sort_by_key(thrustGridIndex, thrustGridIndex+numPhotonsCompact, thrustPhotons);
#else
	thrust::device_ptr<photon> thrustPhotons = thrust::device_pointer_cast(cudaPhotonPool);
	thrust::sort_by_key(thrustGridIndex, thrustGridIndex+numPhotons, thrustPhotons);
#endif

	std::cout<<"Num:"<<numPhotons*numBounces<<" Compacted:"<<numPhotonsCompact<<std::endl;

	// Fill cudaGridFirstPhotonIndex with -1
	int tileSize = 4; //since the grid is 3D
  dim3 photonGridThreadsPerBlock(tileSize, tileSize, tileSize);
  dim3 photonGridBlocksPerGrid((int)ceil(float(grid.xdim)/float(tileSize)), (int)ceil(float(grid.ydim)/float(tileSize)), (int)ceil(float(grid.zdim)/float(tileSize)));
	initGridFirstPhotonIndex<<<photonGridBlocksPerGrid, photonGridThreadsPerBlock>>>(cudaGridFirstPhotonIndex, grid);
	
	// calculate starting photon index for each hashID
#if PHOTONCOMPACT
	computeGridFirstPhoton<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonGridIndex, numPhotonsCompact, cudaGridFirstPhotonIndex);
#else
	computeGridFirstPhoton<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonGridIndex, numPhotons*numBounces, cudaGridFirstPhotonIndex);
#endif

	

	// compress hash-id structure?
}


#define oneOverSqrtTwoPi 0.3989422804f
__device__ float gaussianWeight( float dx, float radius)
{
	float sigma = radius/3.0;
	return (oneOverSqrtTwoPi / sigma) * exp( - (dx*dx) / (2.0 * sigma * sigma) );
}

// Find the index of the photon with the largest distance to the intersection
__device__ int findMaxDistancePhotonIndex(photon* photons, int photonCount, glm::vec3 intersection, int& maxIndex, float& maxDist) {
	maxIndex = -1;
	maxDist = -1;
	for (int i=0; i<photonCount; ++i) {
		photon p = photons[i];
		float dist = glm::distance(p.position, intersection);
		if (dist > maxDist) {
			maxDist = dist;
			maxIndex =  i;
		}
	}
}

// Caculate radiances from photons
__device__ glm::vec3 gatherPhotons(int intersectedGeom, glm::vec3 intersection, glm::vec3 normal, photon* photons, int numTotalPhotons,
															int* gridFirstPhotonIndices, int* gridIndices, gridAttributes grid, float flux) {
	glm::vec3 accumColor(0);

  int px, py, pz; //photon's grid cell index
  getCellIndex(intersection, grid, px, py, pz);
	if (px>=0 && px<grid.xdim && py>=0 && py<grid.ydim && pz>=0 && pz<grid.zdim) { //if intersection is within the grid
		// Find photons in neighboring cells
		photon neighborPhotons[K];
		int neighborPhotonCount = 0;
		for (int i=max(0, px-1); i<min(grid.xdim, px+2); ++i) {
      for (int j=max(0, py-1); j<min(grid.ydim, py+2); ++j) {
				for (int k=max(0, pz-1); k<min(grid.zdim, pz+2); ++k) {
					int gridIndex = gridPhotonHash(i, j, k, grid);
					int firstPhotonIndex = gridFirstPhotonIndices[gridIndex]; //find the index of the first photon in the cell
					if (firstPhotonIndex != -1) {
						int pi = firstPhotonIndex;
						while (pi < numTotalPhotons && gridIndices[pi] == gridIndex) {
							photon p = photons[pi];
							// Check if the photon is on the same geometry as the intersection
							if (p.geomid == intersectedGeom) {
								// We only store K photons. If there are less than K photons stored in the array, add the current photon to the array
								if (neighborPhotonCount < K) {
									neighborPhotons[neighborPhotonCount] = p;
									neighborPhotonCount++;
								}
								// If the array is full, find the photon with the largest distance to the intersection. If current photon's distance
								// to the intersection is smaller, replace the photon with the largest distance with the current photon
								else {
									int maxIndex;
									float maxDist;
									findMaxDistancePhotonIndex(neighborPhotons, K, intersection, maxIndex, maxDist);
									float dist = glm::distance(p.position, intersection);
									if (dist < maxDist) {
										neighborPhotons[maxIndex] = p;
									}
								}
							}
							pi++;
						}
					}
				}
			}
		}

		// Accumulate radiance of the K nearest photons
		for (int i=0; i<neighborPhotonCount; ++i) {
			photon p = neighborPhotons[i];
			float photonDistance = glm::distance(intersection, p.position);
			accumColor += gaussianWeight(photonDistance, RADIUS) * max(0.0f, glm::dot(normal, -p.din)) * p.color;
		}
	}
	return accumColor * flux;
}

// Caculate indirect illumination from photons
__global__ void renderIndirectLighting(glm::vec2 resolution, float time, cameraData cam, glm::vec3* colors, staticGeom* geoms,
																			int numberOfGeoms, triangle* cudafaces, int numFaces, glm::vec3* cudavertices,
																			glm::vec3* cudanormals, glm::vec2* cudauvs, ray* rayPool, photon* photons, int numTotalPhotons, 
																			int* gridFirstPhotonIndices, int* gridIndices, gridAttributes grid, float flux)
{

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if((x<=resolution.x && y<=resolution.y) && glm::length(rayPool[index].transmission) > FLOAT_EPSILON){
		ray r = rayPool[index];        

		//intersection testing
		int intersectedGeom = -1;
		int intersectedMaterial = -1;
		glm::vec3 minIntersectionPoint;
		glm::vec3 minNormal = glm::vec3(0.0f);
		glm::vec2 minUV = glm::vec2(0.0f);

		getClosestIntersection(r, geoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
			minIntersectionPoint, minNormal, intersectedGeom, intersectedMaterial, minUV);

		//Calculate radiance if any geometry is intersected
		if(intersectedGeom > -1)
		{
			colors[index] += gatherPhotons(intersectedGeom, minIntersectionPoint, minNormal, photons, numTotalPhotons, gridFirstPhotonIndices,
				gridIndices, grid, flux);
		}
	}
}

void tracePhotons(int photonThreadsPerBlock, int photonBlocksPerGrid, photon* cudaPhotonPool,
									int numPhotons, staticGeom* cudaGeoms, int numberOfGeoms, material* cudaMaterials,
									cudatexture* cudatextures, float time)
{
	for(int i=0; i <= numBounces; i++)
	{
		// Bounce Photons Around
		bouncePhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons, i,	cudageoms, numGeoms,
			cudafaces, numFaces, cudavertices, cudanormals, cudauvs, cudamaterials, cudatextures, time);

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

void initTexture(cudatexture* textures, float4* cputexturedata, int numberOfTextures, int widthcount, int maxheight) {
	cudaMalloc((void**)&cudatextures, numberOfTextures*sizeof(cudatexture));
	cudaMemcpy(cudatextures, textures, numberOfTextures*sizeof(cudatexture), cudaMemcpyHostToDevice);

	cudaArray* cudatexturedata;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&cudatexturedata, &desc, widthcount, maxheight);
	checkCUDAError("initTexture 1 failed!");
	cudaMemcpyToArray(cudatexturedata, 0, 0, cputexturedata, widthcount * maxheight * sizeof(float4), cudaMemcpyHostToDevice);
	checkCUDAError("initTexture 2 failed!");

	cudaBindTextureToArray(tex, (cudaArray*)cudatexturedata, desc);
	checkCUDAError("initTexture 3 failed!");

	tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = false;
}

//allocate memory for geometry data
void cudaAllocateMemory(int targetFrame, camera* renderCam, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,
												glm::vec3* vertices, int numberOfVertices, glm::vec3* normals, int numberOfNormals, triangle* faces, int numberOfFaces,
												glm::vec2* uvs, int numberOfUVs) {

	int size = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
	
	//send image to GPU
	cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	//convert geometry, transform vertice and normals, send them to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	std::vector<int> lightVec;
	numGeoms = numberOfGeoms;

	float totalEmittance = 0.0f;

	numFaces = numberOfFaces;
	numVertices = numberOfVertices;
	numNormals = numberOfNormals;
	numUVs = numberOfUVs;

	//create arrays for transformed vertices, normals
	glm::vec3* transVertices = new glm::vec3[numberOfVertices];
	glm::vec3* transNormals = new glm::vec3[numberOfNormals];

	int transVCount = 0; //number of vertices transformed
	int transNCount = 0; //number of normals transformed

	//for finding the grid data
	glm::vec3 gridMin(100000);
	glm::vec3 gridMax(-100000);

  for(int i=0; i<numberOfGeoms; ++i){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[targetFrame];
		newStaticGeom.rotation = geoms[i].rotations[targetFrame];
		newStaticGeom.scale = geoms[i].scales[targetFrame];
		newStaticGeom.transform = geoms[i].transforms[targetFrame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[targetFrame];
		buildAABB(newStaticGeom);		//find the bounding box for this geometry
		geomList[i] = newStaticGeom;
		
		//find the corners of biggest bounding box
		gridMin = glm::min(gridMin, newStaticGeom.boundingBox.xyzMin);
		gridMax = glm::max(gridMax, newStaticGeom.boundingBox.xyzMax);

		if (geoms[i].type == MESH) {
			// transform vertices
			for (int j=transVCount; j<transVCount+geoms[i].vertexcount; ++j) {
				transVertices[j] = multiplyMV(geoms[i].transforms[targetFrame], glm::vec4(vertices[j], 1.0f));
			}
			transVCount += geoms[i].vertexcount;

			// transform normals
			for (int j=transNCount; j<transNCount+geoms[i].normalcount; ++j) {
				transNormals[j] = glm::normalize(multiplyMV(getNormalTransform(geomList[i].transform), glm::vec4(normals[j], 0.0f)));
			}
			transNCount += geoms[i].normalcount;
		}

		//store which objects are lights
		if(materials[geoms[i].materialid].emittance > 0)
		{
			lightVec.push_back(i);
			// Add surface area of light here too?
			totalEmittance += materials[geoms[i].materialid].emittance;
		}
  }

	//update grid size
	grid.xmin = gridMin.x; grid.ymin = gridMin.y; grid.zmin = gridMin.z;
	grid.xmax = gridMax.x; grid.ymax = gridMax.y; grid.zmax = gridMax.z;
	grid.xdim = abs(gridMax.x - gridMin.x);
	grid.ydim = abs(gridMax.y - gridMin.y);
	grid.zdim = abs(gridMax.z = gridMin.z);

	totalEnergy = totalEmittance * emitEnergyScale;

	//copy geoms to memory
	cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	//copy mesh data to memory
	cudavertices = NULL;
	cudaMalloc((void**)&cudavertices, numberOfVertices*sizeof(glm::vec3));
	cudaMemcpy(cudavertices, transVertices, numberOfVertices*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudanormals = NULL;
	cudaMalloc((void**)&cudanormals, numberOfNormals*sizeof(glm::vec3));
	cudaMemcpy(cudanormals, transNormals, numberOfNormals*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudafaces = NULL;
	cudaMalloc((void**)&cudafaces, numberOfFaces*sizeof(triangle));
	cudaMemcpy(cudafaces, faces, numberOfFaces*sizeof(triangle), cudaMemcpyHostToDevice);
	cudauvs = NULL;
	cudaMalloc((void**)&cudauvs, numberOfUVs*sizeof(glm::vec2));
	cudaMemcpy(cudauvs, uvs, numberOfUVs*sizeof(glm::vec2), cudaMemcpyHostToDevice);

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

	//Create Memory for RayPool
	cudaMalloc((void**)&cudarays, (renderCam->resolution.x * renderCam->resolution.y) * sizeof(ray));

#if COMPACTION
	cudaMalloc((void**)&cudarays2, (renderCam->resolution.x * renderCam->resolution.y) * sizeof(ray));
#endif

	//std::cout<<"allocating mem for photon pool"<<std::endl;
	cudaPhotonPool = NULL;
	cudaMalloc((void**)&cudaPhotonPool, numBounces * numPhotons * sizeof(photon));

#if PHOTONCOMPACT
	cudaPhotonPoolCompact = NULL;
	cudaMalloc((void**)&cudaPhotonPoolCompact, numBounces * numPhotons * sizeof(photon));
#endif

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

	//set up data of grid
	// Compute photon grid dimensions
	grid.xdim = ceil((grid.xmax - grid.xmin)/grid.cellsize);
	grid.ydim = ceil((grid.ymax - grid.ymin)/grid.cellsize);
	grid.zdim = ceil((grid.zmax - grid.zmin)/grid.cellsize);

	cudaAllocateAccumulatorImage(renderCam);

	cudaGridFirstPhotonIndex=NULL;
	cudaMalloc((void**)&cudaGridFirstPhotonIndex, grid.xdim * grid.ydim * grid.zdim * sizeof(int));

	delete[] geomList;
	delete[] lightID;
	delete[] transVertices;
	delete[] transNormals;
}

void cudaFreeTexture() {
	cudaUnbindTexture(tex);
	cudaFree(cudatextures);
	cudaFree(cudatexturedata);
}

//free up memory
void cudaFreeMemory() {

	cudaFree( cudaimage);
	cudaFree( cudageoms );
	cudaFree( cudafaces );
	cudaFree( cudavertices );
	cudaFree( cudanormals );
	cudaFree( cudamaterials);
	cudaFree( cudaLights);

	cudaFree(cudarays);
#if COMPACTION
	cudaFree(cudarays2);
#endif

	cudaFree(cudaPhotonPool);
#if PHOTONCOMPACT
	cudaFree(cudaPhotonPoolCompact);
#endif
	cudaFree(cudaGridFirstPhotonIndex);

	cudaFreeAccumulatorImage();

}

void cudaAllocateIterationMemory(camera* renderCam)
{
	// cudaPhotonGridIndex is allocated only when needed
}

void cudaFreeIterationMemory()
{
	//free up stuff, or else we'll leak memory like a madman

	//free grid data
	if(cudaPhotonGridIndex)
	{
		cudaFree(cudaPhotonGridIndex);
		cudaPhotonGridIndex = NULL;
	}
}

void cudaPhotonMapCore(camera* renderCam, int frame, int iterations, uchar4* PBOpos, cameraData liveCamera)
{
	cudaAllocateIterationMemory(renderCam);

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

	// Generate Photon List
	int photonThreadsPerBlock = 512;
	int photonBlocksPerGrid = ceil(numPhotons * 1.0f/photonThreadsPerBlock);

	emitPhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons, numBounces, cudageoms, 
		cudaLights, numLights, cudaAccumLightProbabilities, cudamaterials, iterations);
	cudaThreadSynchronize();
	checkCUDAError("emit photons kernel failed!");

	// Trace all photons with all bounces
	tracePhotons(photonThreadsPerBlock, photonBlocksPerGrid, cudaPhotonPool, numPhotons, cudageoms, numGeoms, cudamaterials, cudatextures, iterations);
	cudaThreadSynchronize();
	checkCUDAError("tracePhotons kernel failed!");

	// Assume each light emits the same number of photons, calculate the flux per photon
	flux = totalEnergy/(float)numPhotons;

	buildSpatialHash();
	cudaThreadSynchronize();
	checkCUDAError("calculating grid idx kernel failed!");
}

//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
														staticGeom* geoms, int numberOfGeoms, triangle* cudafaces, int numFaces, glm::vec3* cudavertices,
														glm::vec3* cudanormals, glm::vec2* cudauvs, material* materials, cudatexture* cudatextures,
														photon* photons, int numTotalPhotons, int* gridFirstPhotonIndices, int* gridIndices,
														int mode, gridAttributes grid, float flux, ray* rayPool
#if COMPACTION
														, int numberOfRays
#endif
														)
{
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

		//intersection testing
		int intersectedGeom = -1;
		int intersectedMaterial = -1;
		glm::vec3 minIntersectionPoint;
		glm::vec3 minNormal = glm::vec3(0.0f);
		glm::vec2 minUV = glm::vec2(0.0f);

		getClosestIntersection(r, geoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
			minIntersectionPoint, minNormal, intersectedGeom, intersectedMaterial, minUV);

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
			diffuseColor = getMaterialColor(m, cudatextures, minUV);
			specularColor = m.specularColor;
			// Emmited color is the same as material color
			emittance = m.color * m.emittance;

			if (mode != DISP_PATHTRACE) {
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

						bool visible = visibilityCheck(shadowRay,geoms,numberOfGeoms,cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
							lightSourceSample,iter);

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
			}

			AbsorptionAndScatteringProperties absScatProps;
			glm::vec3 colorSend, unabsorbedColor;
			ray returnRay = r;
			int rayPropogation = calculateBSDF(returnRay,minIntersectionPoint,minNormal,diffuseColor*m.emittance,absScatProps,colorSend,unabsorbedColor,m,
												resolution, time, rayDepth, x, y);

			// Compute direct illumination
			glm::vec3 surfaceColor = emittance + diffuseLight * diffuseColor +  phongLight * specularColor;
			// Compute indirect illumination if we are using photon map
			if (mode == DISP_COMBINED) {
				surfaceColor += gatherPhotons(intersectedGeom, minIntersectionPoint, minNormal, photons, numTotalPhotons, gridFirstPhotonIndices, gridIndices, grid, flux);
			}

			// Diffuse Reflection or light source
			if (rayPropogation == 0)
			{
				if (mode == DISP_PATHTRACE) {
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
				}
				else {
#if COMPACTION
					colors[r.pixelIndex] += r.transmission * surfaceColor;
#else
					colors[index] += r.transmission * surfaceColor;
#endif
					r.transmission = glm::vec3(0);
				}
				rayPool[index] = r;

			}
			// Reflection; calculate transmission coeffiecient
			else if(rayPropogation == 1)
			{
				if (mode == DISP_PATHTRACE) {
#if COMPACTION
					colors[r.pixelIndex] += r.transmission * emittance;
#else
					colors[index] += r.transmission * emittance;
#endif
				}
				else {
#if COMPACTION
					colors[r.pixelIndex] += r.transmission * (1.0f - m.hasReflective) * surfaceColor;
#else
					colors[index] += r.transmission * (1.0f - m.hasReflective) * surfaceColor;
#endif
				}
				returnRay.transmission = r.transmission * diffuseColor *  m.hasReflective;
				rayPool[index] = returnRay;
			}
			// Refraction; calculate transmission coeffiecient
			else if (rayPropogation == 2)
			{
				if (mode == DISP_PATHTRACE) {
#if COMPACTION
					colors[r.pixelIndex] += r.transmission * emittance;
#else
					colors[index] += r.transmission * emittance;
#endif
				}
				else {
#if COMPACTION
					colors[r.pixelIndex] += r.transmission * (1.0f - m.hasRefractive) * surfaceColor;
#else
					colors[index] += r.transmission * (1.0f - m.hasRefractive) * surfaceColor;
#endif
				}
				returnRay.transmission = r.transmission * diffuseColor * m.hasRefractive;
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

void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, cameraData liveCamera){

  cudaAllocateIterationMemory(renderCam);

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

  //clear On screen buffer
  clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution, cudaimage);

	//display photons mode
	if (mode == DISP_PHOTONS)
	{
		// Calculate Viewport * Projection * View matrix from camera info
		glm::vec3 center = cam.position + cam.view;

		glm::mat4 viewMat = glm::lookAt(cam.position, center, cam.up);
		glm::mat4 projectionMat = glm::perspective(cam.fov.y*2, cam.resolution.x/cam.resolution.y, 0.1f, 1000.0f);
		cudaMat4 viewProjectionViewPort = utilityCore::glmMat4ToCudaMat4(projectionMat*viewMat);

		int photonThreadsPerBlock = 512;
		int photonBlocksPerGrid = ceil(numPhotons * 1.0f/photonThreadsPerBlock);

		// Display all photons in the photonImage buffer
#if PHOTONCOMPACT 
		photonBlocksPerGrid = ceil(numPhotonsCompact * 1.0f/photonThreadsPerBlock);

		displayPhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPoolCompact, numPhotonsCompact, cam.resolution, 
			cam, cudaimage, viewProjectionViewPort, flux);
#else
		photonBlocksPerGrid = ceil(numPhotons*numBounces * 1.0f/photonThreadsPerBlock);
		displayPhotons<<<dim3(photonBlocksPerGrid),dim3(photonThreadsPerBlock)>>>(cudaPhotonPool, numPhotons * numBounces, resolution, 
			cam, cudaimage, viewProjectionViewPort, flux);
#endif

		cudaThreadSynchronize();
		checkCUDAError("display photons kernel failed!");
	}	

	//ray trace or photon map mode
	else {
		//Fill ray pool with rays from camera for first iteration
		fillRayPoolFromCamera<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudarays);
		int numberOfRays = (int)renderCam->resolution.x * (int)renderCam->resolution.y;

		//Render indirect illumination from photons
		if (mode == DISP_GATHER) {
#if PHOTONCOMPACT 
			renderIndirectLighting<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudaimage, cudageoms, numGeoms,
				cudafaces, numFaces, cudavertices, cudanormals, cudauvs, cudarays, cudaPhotonPoolCompact, numPhotonsCompact,
				cudaGridFirstPhotonIndex, cudaPhotonGridIndex, grid, flux);
#else
			renderIndirectLighting<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, cudaimage, cudageoms, numGeoms,
				cudafaces, numFaces, cudavertices, cudanormals, cudauvs, cudarays, cudaPhotonPool, numPhotons * numBounces,
				cudaGridFirstPhotonIndex, cudaPhotonGridIndex, grid, flux);
#endif

			cudaThreadSynchronize();
			checkCUDAError("gather photons kernel failed!");
		}
		else {
			int linearTileSize = tileSize*tileSize;
			for(int i=0; i < MAX_RECURSION_DEPTH && numberOfRays > 0; i++)
			{
#if COMPACTION
				dim3 linearGridSize((int)ceil(numberOfRays*1.0f/linearTileSize),1,1);
#if PHOTONCOMPACT
				raytraceRay<<<linearGridSize, dim3(linearTileSize,1,1)>>>(renderCam->resolution, (float)iterations, cam, traceDepth+i,
					cudaimage, cudageoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
					cudamaterials, cudatextures, cudaPhotonPoolCompact, numPhotonsCompact,
					cudaGridFirstPhotonIndex, cudaPhotonGridIndex, mode, grid, flux,
					i%2==0?cudarays : cudarays2,
					numberOfRays);
#else
				raytraceRay<<<linearGridSize, dim3(linearTileSize,1,1)>>>(renderCam->resolution, (float)iterations, cam, traceDepth+i,
					cudaimage, cudageoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
					cudamaterials, cudatextures, cudaPhotonPool, numPhotons * numBounces,
					cudaGridFirstPhotonIndex, cudaPhotonGridIndex, mode, grid, flux,
					i%2==0?cudarays : cudarays2,
					numberOfRays);
#endif
				checkCUDAError("Ray Trace Failed!");	 

				numberOfRays = streamCompactRayPool( i%2==0? cudarays : cudarays2,
					i%2==0? cudarays2 : cudarays,
					numberOfRays);
#else
#if PHOTONCOMPACT
				raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth+i, cudaimage,
					cudageoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
					cudamaterials, cudatextures, cudaPhotonPoolCompact, numPhotonsCompact, cudaGridFirstPhotonIndex,
					cudaPhotonGridIndex, mode, grid, flux, cudarays);
#else
				raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth+i, cudaimage,
					cudageoms, numberOfGeoms, cudafaces, numFaces, cudavertices, cudanormals, cudauvs,
					cudamaterials, cudatextures, cudaPhotonPool, numPhotons * numBounces, cudaGridFirstPhotonIndex,
					cudaPhotonGridIndex, mode, grid, flux, cudarays);
#endif
#endif
			}
		}
	}

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

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");

  cudaFreeIterationMemory();
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