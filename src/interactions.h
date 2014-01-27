// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define FRESNEL 1
#define SCHLICK 0

#include "intersections.h"

struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient;
};

//forward declaration
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  Fresnel fresnel;

  fresnel.reflectionCoefficient = 1;
  fresnel.transmissionCoefficient = 0;
  return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    //Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    //Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation.
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {

	float theta = 2.0f * PI * xi1;
	float phi = acos(2.0f * xi2 - 1);

	float radius = 1.0f;

	glm::vec3 localPosition = glm::vec3( radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi), radius * cos(phi));

	return localPosition;
	// No need to normalize since radius is one
	// return glm::normalize(localPosition);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m, 
									   // More variables for random data generation
									   glm::vec2 resolution, float time, float rayDepth, int x, int y){

	// Reflected
	if(fabs(m.hasReflective) > FLOAT_EPSILON)
	{
		ray reflected;
		// reflected.direction = glm::reflect(r.direction,normal);
		reflected.direction = r.direction - 2.0f * normal  * glm::dot(normal,r.direction);
		reflected.origin = intersect + NUDGE * reflected.direction;
		r = reflected;
		return 1;
	}
	// Transmitted
	else if(fabs(m.hasRefractive) > FLOAT_EPSILON)
	{
		ray refracted;
		
		float dotValue = glm::dot(r.direction,normal);
		float IOR = 1.0f/m.indexOfRefraction;
		if(dotValue > 0)
		{
			IOR = 1.0f/IOR;
			normal = -normal;
		}
		float cosValue = -glm::dot(r.direction,normal);
		float k = 1 - IOR*IOR * (1 - cosValue * cosValue);

		// Total Internal Reflection
		if(k < 0)
		{
			refracted.direction = r.direction - 2.0f * normal  * glm::dot(normal,r.direction);
			refracted.origin = intersect + NUDGE * refracted.direction;
			r = refracted;
			return 2;
		}

		refracted.direction = glm::normalize(r.direction * IOR + normal * (IOR * cosValue - sqrt(k)));
		refracted.origin = intersect + 0.01f * refracted.direction;

#if FRESNEL
		// Fresnel Calculation
		// Lucy's equations
		float nd = glm::dot(r.direction, normal);
		float nt = glm::dot(refracted.direction, normal);
		float n_a = 1.0f;
		float n_b = 1.0f/IOR;
		
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
		// Fresnel equations
		float reflectedParallel = (n_b * nd - n_a * nt) * (n_b * nd - n_a * nt) / ((n_b * nd + n_a * nt) * (n_b * nd + n_a * nt));
		float reflectedPerpendicular = (n_a * nd - n_b * nt) * (n_a * nd - n_b * nt) / ((n_a * nd + n_b * nt) * (n_a * nd + n_b * nt));
		amountReflected = 0.5 * (reflectedParallel + reflectedPerpendicular);
#endif
		// Stochastically decide whether to reflect or refract
		glm::vec3 randVector = generateRandomNumberFromThread(resolution,time * (rayDepth+1),x,y);

		// If a uniform variable is less than the reflected amount, this ray shall be reflected
		if(randVector.y  < amountReflected)
		{
			refracted.direction = r.direction - 2.0f * normal  * glm::dot(normal,r.direction);
			refracted.origin = intersect + NUDGE * refracted.direction;
		}
#endif

		r = refracted;
		return 2;
	}
	else // Diffuse
	{
		return 0;
	}
};

#endif
