#ifndef KDTREESTRUCTS_H
#define KDTREESTRUCTS_H

#include "glm/glm.hpp"

struct PlaneGPU
{
	int axis;
	float splitPoint;
};

struct KDNodeGPU
{
	int first, second;
	int ropes[6];
	int startPrimIndex;
	int numPrims;
	glm::vec3 llb;
	glm::vec3 urf;
	PlaneGPU splitPlane;

};

#endif