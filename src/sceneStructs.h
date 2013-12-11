// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

#define COMPACTION 1

struct triangle {
	int geomid;
	int v1;
	int v2;
	int v3;
	int n1;
	int n2;
	int n3;

	//texture coordinates
	int t1;
	int t2;
	int t3;

	triangle() {};
	triangle(int id, int vi1, int vi2, int vi3, int ni1, int ni2, int ni3, int ti1, int ti2, int ti3) :
		geomid(id), v1(vi1), v2(vi2), v3(vi3), n1(ni1), n2(ni2), n3(ni3), t1(ti1), t2(ti2), t3(ti3) {};
};

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;

#if COMPACTION
	int pixelIndex;
#endif

	glm::vec3 transmission;

};

struct photon {
	bool stored;			// Check if the current photon has been stored, for gather
	glm::vec3 position;		// Position of photon
	glm::vec3 din;	// Incoming direction of photon (potentially use theta phi?)
	glm::vec3 dout; // Outgoing direction
	glm::vec3 color;		// I think this essentially stores flux as well. : change to spectral eventually?
	short geomid; // Which surface the photon is on, -1 means no intersection
#if COMPACTION
	int originalIndex;		// If using stream compaction
#endif
};

struct gridAttributes {
	float xmin, ymin, zmin;
	float xmax, ymax, zmax;
	float cellsize;
	int xdim, ydim, zdim;

	gridAttributes(float x1, float y1, float z1, float x2, float y2, float z2, float cs) {
		xmin = x1; ymin = y1; zmin = z1;
		xmax = x2; ymax = y2; zmax = z2;
		cellsize = cs;
	};
};


struct geom {
	enum GEOMTYPE type;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;

	// for mesh
	int vertexcount;
	int normalcount;
	int facecount;
};

//axis aligned bounding box
struct AABB {
	
	//min and max positions of bounding box
	glm::vec3 xyzMin;
	glm::vec3 xyzMax;
	glm::vec3 dimension; 

};

struct staticGeom {
	enum GEOMTYPE type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
	AABB boundingBox;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	float aperture;
	float focusPlane;
};

struct camera {
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int frames;
	glm::vec2 fov;
	float* apertures;
	float* focusPlanes;
	unsigned int iterations;
	glm::vec3* image;
	ray* rayList;
	std::string imageName;
};

struct material{
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
	int textureid; // -1 means no texture
};

// texture structure on CPU
struct cputexture {
	int width;
	int height;
	glm::vec3* colors;
};

// texture structure on CUDA
struct cudatexture {
	int width;
	int height;
	int xindex; // the x index of the first pixel in the texture object with all textures
};

#endif //CUDASTRUCTS_H
