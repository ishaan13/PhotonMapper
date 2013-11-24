// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//multiplication that takes into account w for perspective projection
__host__ __device__ glm::vec4 multiplyMV_4(cudaMat4 m, glm::vec4 v){
        glm::vec4 r(1);
        r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
        r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
        r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
        r.w = (m.w.x*v.x)+(m.w.y*v.y)+(m.w.z*v.z)+(m.w.w*v.w);

        return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	glm::vec3 ro = multiplyMV(box.inverseTransform,glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize( multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)) );

	ray rt; rt.origin = ro; rt.direction = rd;

	float halfWidth = 0.5f;

	
	// Now box is in its own local unit space 
	glm::vec3 inverseDirection;

	glm::vec3 tmin, tmax;

	inverseDirection = 1.0f / rd;

	// min max x
	if(inverseDirection.x >= 0)
	{
		tmin.x = (-halfWidth - rt.origin.x) * inverseDirection.x;
		tmax.x = ( halfWidth - rt.origin.x) * inverseDirection.x;
	}
	else
	{
		tmin.x = ( halfWidth - rt.origin.x) * inverseDirection.x;
		tmax.x = (-halfWidth - rt.origin.x) * inverseDirection.x;
	}


	// min max y
	if(inverseDirection.y >= 0)
	{
		tmin.y = (-halfWidth - rt.origin.y) * inverseDirection.y;
		tmax.y = ( halfWidth - rt.origin.y) * inverseDirection.y;
	}
	else
	{
		tmin.y = ( halfWidth - rt.origin.y) * inverseDirection.y;
		tmax.y = (-halfWidth - rt.origin.y) * inverseDirection.y;
	}

	if( (tmin.x > tmax.y) || (tmin.y > tmax.x)) return -1;
	if( tmin.y > tmin.x ) tmin.x = tmin.y;
	if( tmax.y < tmax.x ) tmax.x = tmax.y;

	// min max z
	if(inverseDirection.z >= 0)
	{
		tmin.z = (-halfWidth - rt.origin.z) * inverseDirection.z;
		tmax.z = ( halfWidth - rt.origin.z) * inverseDirection.z;
	}
	else
	{
		tmin.z = ( halfWidth - rt.origin.z) * inverseDirection.z;
		tmax.z = (-halfWidth - rt.origin.z) * inverseDirection.z;
	}

	if( (tmin.x > tmax.z) || (tmin.z > tmax.x)) return -1;
	if( tmin.z > tmin.x ) tmin.x = tmin.z;
	if( tmax.z < tmax.x ) tmax.x = tmax.z;

	float distanceLocal = glm::min(tmin.x,tmax.x);

	glm::vec3 localIntersectionPoint = getPointOnRay(rt, distanceLocal);

	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(localIntersectionPoint, 1.0));
    
	intersectionPoint = realIntersectionPoint;
	glm::vec3 localNormal;

	// Calculating Normal
	if(fabs(localIntersectionPoint.x - -halfWidth) < FLOAT_EPSILON)
	{
		localNormal = glm::vec3(-1,0,0);
	}
	else if( fabs(localIntersectionPoint.x - halfWidth) < FLOAT_EPSILON)
	{
		localNormal = glm::vec3(1,0,0);
	}
	else if(fabs(localIntersectionPoint.y - -halfWidth) < FLOAT_EPSILON)
	{
		localNormal = glm::vec3(0,-1,0);
	}
	else if( fabs(localIntersectionPoint.y - halfWidth) < FLOAT_EPSILON)
	{
		localNormal = glm::vec3(0,1,0);
	}
	else if(fabs(localIntersectionPoint.z - -halfWidth) < FLOAT_EPSILON)
	{
		localNormal = glm::vec3(0,0,-1);
	}
	else if( fabs(localIntersectionPoint.z - halfWidth) < FLOAT_EPSILON)
	{
		localNormal = glm::vec3(0,0,1);
	}


	// Psuedo point is one local unit distance behind the local intersection point in the direction of the normal
	glm::vec3 realPseudoPoint = multiplyMV(box.transform, glm::vec4(localIntersectionPoint - localNormal ,1.0f));
	normal = glm::normalize(realIntersectionPoint - realPseudoPoint);
    
	// @DO: something is not normalized. Confirm what
	//return glm::length(r.origin - realIntersectionPoint);
	//return glm::length(r.origin - realIntersectionPoint) * glm::length(r.direction);

	return distanceLocal;

	

	/*
	560- ray tracer implementation
	
	float tNear = -1000000, tFar = 10000000;
	float t1,t2;

	float returnT;

	for(int i=0; i<3;i++)
	{
		float xD, xo;
		switch(i)
		{
			case 0: xD = rt.direction.x; xo = rt.origin.x; break;
			case 1: xD = rt.direction.y; xo = rt.origin.y; break;
			case 2: xD = rt.direction.z; xo = rt.origin.z; break;
		}
		if(fabs(xD)<0.00001)
		{
			if(xo < -halfWidth || xo > halfWidth) 
				return -1;	// Handle Parallel Case
		}
		t1 = (-halfWidth - xo)/xD; //(-0.5 - xo)/xD; // my implementation has -1 to 1
		t2 = ( halfWidth - xo)/xD; //( 0.5 - xo)/xD; // my implementation has -1 to 1


		//@DO: is this how insides are checked?
		if ( t1 > t2)// && !(t2<0)) 
		{
			float temp = t2;
			t2 = t1;
			t1 = temp;
		}
		if (t1 > tNear )
			tNear = t1;
		if (t2 < tFar)// && t2>0)
			tFar = t2;
		if( tNear > tFar)
			return -1;//Box missed, do something
		if( tFar < 0 )
			return -1;//Box behind ray, do something

		if(tNear < 0)
			returnT = tFar;
		else
			returnT = tNear;
	}
	//cout<<tNear<<endl;
	glm::vec3 position = rt.origin + returnT * rt.direction;
	glm::vec4 tNrm;

	if(fabs(position.x - halfWidth)< FLOAT_EPSILON)
		tNrm = glm::vec4(1,0,0,0);
	else if(fabs(position.x - (-halfWidth))< FLOAT_EPSILON)
		tNrm = glm::vec4(-1,0,0,0);
	else if(fabs(position.y - halfWidth)< FLOAT_EPSILON)
		tNrm = glm::vec4(0,1,0,0);
	else if(fabs(position.y - (-halfWidth))< FLOAT_EPSILON)
		tNrm = glm::vec4(0,-1,0,0);
	else if(fabs(position.z - halfWidth)< FLOAT_EPSILON)
		tNrm = glm::vec4(0,0,1,0);
	else if(fabs(position.z - (-halfWidth))< FLOAT_EPSILON)
		tNrm =glm:: vec4(0,0,-1,0);
	
	intersectionPoint = multiplyMV(box.transform,glm::vec4(position,1.0f));
	normal = glm::normalize(multiplyMV(box.transform,tNrm));

	return returnT;
	*/
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
}

// ray-plane intersection for 
__host__ __device__ float planeIntersectionTest(glm::vec3 pointOnPlane, glm::vec3 normalOfPlane, ray r, glm::vec3 &intersection)
{
	float denominator = glm::dot(r.direction,normalOfPlane);
	if(fabs(denominator) < FLOAT_EPSILON)
	{
		return -1;
	}
	float numerator = glm::dot((pointOnPlane - r.origin),normalOfPlane);
	float distance = numerator/denominator;
	intersection = r.origin + distance * r.direction;
	return distance;
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
// Marsaglia (1972)
// http://mathworld.wolfram.com/SpherePointPicking.html
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

	/*
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> uniformDistribution(-1,1);
	float x1, x2;
	int flag = 0;
	while(flag == 0)
	{
		x1 = (float) uniformDistribution(rng);
		x2 = (float) uniformDistribution(rng);

		if( (x1*x1 + x2*x2) < 1.0f)
			flag = 1;
	}

	float term1 = x1*x1 + x2*x2;
	float term2 = sqrt(1 - term1);
	
	glm::vec3 unitSphere = glm::vec3( 2 * x1 * term2,
							2 * x2 * term2,
							1 - 2 * term1	);
	
	*/

	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> uniformDistribution(0,1);
	float x1, x2;
	x1 = (float) uniformDistribution(rng);
	x2 = (float) uniformDistribution(rng);

	float theta = 2.0f * PI * x1;
	float phi = acos(2.0f * x2 - 1);

	float radius = 0.5f;

	glm::vec3 localPosition = glm::vec3( radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi));

	glm::vec3 realPosition = multiplyMV(sphere.transform, glm::vec4(localPosition,1.0f));

	return realPosition;
}

#endif


