#include "intersections.h"
#include "KDTree.h"
#include "glm/gtc/matrix_transform.hpp"

enum {
	KD_ON,
	KD_OFF
};

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

__host__ __device__ cudaMat4 getNormalTransform(cudaMat4 a){
	glm::mat4 m = utilityCore::cudaMat4ToGlmMat4(a);
	m = glm::inverse(glm::transpose(m));
	return utilityCore::glmMat4ToCudaMat4(m);
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

__host__ __device__ glm::vec3 getNormalOfPointOnUnitCube(glm::vec3 point) {
	float halfWidth = 0.5f;
	glm::vec3 normal;

	if(fabs(point.x - -halfWidth) < FLOAT_EPSILON)
	{
		normal = glm::vec3(-1,0,0);
	}
	else if( fabs(point.x - halfWidth) < FLOAT_EPSILON)
	{
		normal = glm::vec3(1,0,0);
	}
	else if(fabs(point.y - -halfWidth) < FLOAT_EPSILON)
	{
		normal = glm::vec3(0,-1,0);
	}
	else if( fabs(point.y - halfWidth) < FLOAT_EPSILON)
	{
		normal = glm::vec3(0,1,0);
	}
	else if(fabs(point.z - -halfWidth) < FLOAT_EPSILON)
	{
		normal = glm::vec3(0,0,-1);
	}
	else if( fabs(point.z - halfWidth) < FLOAT_EPSILON)
	{
		normal = glm::vec3(0,0,1);
	}

	return normal;
}

__host__ __device__ glm::vec2 getUVOfPointOnUnitCube(glm::vec3 point) {
	float halfWidth = 0.5f;
	glm::vec2 uv;

	if(fabs(point.x - -halfWidth) < FLOAT_EPSILON)
	{
		uv = glm::vec2(-point.z+0.5, -point.y+0.5);
	}
	else if( fabs(point.x - halfWidth) < FLOAT_EPSILON)
	{
		uv = glm::vec2(point.z+0.5, -point.y+0.5);
	}
	else if(fabs(point.y - -halfWidth) < FLOAT_EPSILON)
	{
		uv = glm::vec2(point.x+0.5, point.z+0.5);
	}
	else if( fabs(point.y - halfWidth) < FLOAT_EPSILON)
	{
		uv = glm::vec2(point.x+0.5, -point.z+0.5);
	}
	else if(fabs(point.z - -halfWidth) < FLOAT_EPSILON)
	{
		uv = glm::vec2(-point.x+0.5, -point.y+0.5);
	}
	else if( fabs(point.z - halfWidth) < FLOAT_EPSILON)
	{
		uv = glm::vec2(-point.x+0.5, -point.y+0.5);
	}

	return uv;
}

__device__ float getClosestIntersection(ray r, staticGeom* geoms, int numberOfGeoms, triangle* faces, int numberOfFaces, glm::vec3* vertices,
	glm::vec3* normals, glm::vec2* uvs, glm::vec3& minIntersectionPoint, glm::vec3& minNormal,
	int& intersectedGeom, int& intersectedMaterial, glm::vec2& minUV, KDNodeGPU* cudakdtree, int treeRootIndex, int* cudaPrimIndex, int kdmode) {
		float minDepth = FLT_MAX;

		float heatValue = 0.0f;

		for (int iter=0; iter < numberOfGeoms; iter++)
		{
			float depth=-1;
			glm::vec3 intersection;
			glm::vec3 normal;
			staticGeom currentGeometry = geoms[iter];
			glm::vec2 uv;

			if (currentGeometry.type == CUBE)
			{
				depth = boxIntersectionTest(currentGeometry,r,intersection,normal,uv);
			}

			else if (geoms[iter].type == SPHERE)
			{
				depth = sphereIntersectionTest(currentGeometry,r,intersection,normal,uv);
			}

			if (depth > 0 && depth < minDepth)
			{
				minDepth = depth;
				minIntersectionPoint = intersection;
				minNormal = normal;
				intersectedGeom = iter;
				intersectedMaterial = currentGeometry.materialid;
				minUV = uv;
			}
		}

		if (kdmode == KD_ON) {
			glm::vec3 intersection;
			glm::vec3 normal;
			glm::vec2 uv;
			int geomid;
			int mtlid;
			float depth = traverse(r, cudakdtree, treeRootIndex, vertices, normals, uvs, faces, cudaPrimIndex, geoms, intersection, normal, geomid, mtlid, uv, heatValue);

			if (depth > 0.0001f && depth < minDepth) {
				minDepth = depth;
				minIntersectionPoint = intersection;
				minNormal = normal;
				intersectedGeom = geomid;
				intersectedMaterial = mtlid;
				minUV = uv;
			}
		}
		else {
			// get closest intersection with triangles
			for (int i=0; i<numberOfFaces; ++i) {
				glm::vec3 v1 = vertices[faces[i].v1];
				glm::vec3 v2 = vertices[faces[i].v2];
				glm::vec3 v3 = vertices[faces[i].v3];
				glm::vec3 n1 = normals[faces[i].n1];
				glm::vec3 n2 = normals[faces[i].n2];
				glm::vec3 n3 = normals[faces[i].n3];
				glm::vec2 t1 = uvs[faces[i].t1];
				glm::vec2 t2 = uvs[faces[i].t2];
				glm::vec2 t3 = uvs[faces[i].t3];
				glm::vec3 intersection;
				glm::vec3 normal;
				glm::vec2 uv;
				float depth = triangleIntersectionTest(v1, v2, v3, n1, n2, n3, t1, t2, t3, r, intersection, normal, uv);
				if (depth > 0 && depth < minDepth) {
					minDepth = depth;
					minIntersectionPoint = intersection;
					minNormal = normal;
					intersectedGeom = faces[i].geomid;
					intersectedMaterial = geoms[intersectedGeom].materialid;
					minUV = uv;
				}
			}
		}

		return heatValue;
}

__device__ bool visibilityCheck(ray r, staticGeom* geoms, int numberOfGeoms, triangle* faces, int numberOfFaces, glm::vec3* vertices,
	glm::vec3* normals, glm::vec2* uvs, glm::vec3 pointToCheck, int lightSourceIndex, KDNodeGPU* cudakdtree, int treeRootIndex, int* cudaPrimIndex, int kdmode)
{
	bool visible = true;
	float distance = glm::length(r.origin - pointToCheck);
	float minDepth = FLT_MAX;
	int nearestGeom = -1;

	// Check whether any object occludes point to check from ray's origin
	for(int iter=0; iter < numberOfGeoms; iter++)
	{
		/*
		// Avoid calculating self intersections
		if(iter==lightSourceIndex)
			continue;
		*/

		float depth=-1;
		glm::vec3 intersection;
		glm::vec3 normal;
		glm::vec2 uv;

		if(geoms[iter].type == CUBE)
		{
			depth = boxIntersectionTest(geoms[iter],r,intersection,normal,uv);
		}

		else if(geoms[iter].type == SPHERE)
		{
			depth = sphereIntersectionTest(geoms[iter],r,intersection,normal,uv);
		}

		if(depth > 0 && depth < minDepth)
		{
			minDepth = depth;
			nearestGeom = iter;
		}
		if(iter != lightSourceIndex && (depth > 0 && (depth + NUDGE) < distance))
		{
			//printf("Depth: %f\n", depth);
			return false;
		}
	}

	// get closest intersection with triangles
	if (kdmode == KD_ON) {
		glm::vec3 intersection;
		glm::vec3 normal;
		glm::vec2 uv;
		int geomid;
		int mtlid;

		float temp;
		float depth = traverse(r, cudakdtree, treeRootIndex, vertices, normals, uvs, faces, cudaPrimIndex, geoms, intersection, normal, geomid, mtlid, uv, temp);
		
		if(depth > 0 && depth < minDepth)
		{
			minDepth = depth;
			nearestGeom = geomid;
		}
		if (geomid!= lightSourceIndex && (depth > 0 && (depth + NUDGE) < distance)) {
			return false;
		}

	}
	else {
		for (int i=0; i<numberOfFaces; ++i) {

			// Reduce global memory lookups
			triangle tri = faces[i];

			glm::vec3 v1 = vertices[tri.v1];
			glm::vec3 v2 = vertices[tri.v2];
			glm::vec3 v3 = vertices[tri.v3];
			glm::vec3 n1 = normals[tri.n1];
			glm::vec3 n2 = normals[tri.n2];
			glm::vec3 n3 = normals[tri.n3];
			glm::vec2 t1 = uvs[tri.t1];
			glm::vec2 t2 = uvs[tri.t2];
			glm::vec2 t3 = uvs[tri.t3];
			glm::vec3 intersection;
			glm::vec3 normal;
			glm::vec2 uv;
			float depth = triangleIntersectionTest(v1, v2, v3, n1, n2, n3, t1, t2, t3, r, intersection, normal, uv);
		
			if(depth > 0 && depth < minDepth)
			{
				minDepth = depth;
				nearestGeom = tri.geomid;
			}
			if(tri.geomid != lightSourceIndex && (depth > 0 && (depth + NUDGE) < distance))
			{
				//printf("Depth: %f\n", depth);
				return false;
			}

		}
	}

	if(nearestGeom != lightSourceIndex)
		return false;
	else
		return true;
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv){

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
	glm::vec3 localNormal = getNormalOfPointOnUnitCube(localIntersectionPoint);
	uv = getUVOfPointOnUnitCube(localIntersectionPoint);

	// Psuedo point is one local unit distance behind the local intersection point in the direction of the normal
	glm::vec3 realPseudoPoint = multiplyMV(box.transform, glm::vec4(localIntersectionPoint - localNormal ,1.0f));
	normal = glm::normalize(realIntersectionPoint - realPseudoPoint);

	return distanceLocal;
}


//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv){

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

	glm::vec3 localPoint = getPointOnRay(rt, t);

	glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(localPoint, 1.0));
	glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

	intersectionPoint = realIntersectionPoint;
	normal = glm::normalize(realIntersectionPoint - realOrigin);

	uv = glm::vec2(0.5 + atan2(localPoint.z, localPoint.x) / (2.0f * PI), 0.5 - asin(localPoint.y) / PI);

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

__host__ __device__ float triangleIntersectionTest(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3,
	glm::vec2 t1, glm::vec2 t2, glm::vec2 t3, ray r, glm::vec3& intersection, glm::vec3& normal, glm::vec2& uv){
		if (glm::length(v1 - v2) < EPSILON || glm::length(v1 - v3) < EPSILON || glm::length(v2 - v3) < EPSILON)
			return -1;

		// ray plane intersection
		glm::vec3 n = glm::cross(v2 - v1, v3 - v1);
		float t = -glm::dot(r.origin - v1, n) / glm::dot(r.direction, n);
		if (t <= 0) {
			return -1;
		}
		glm::vec3 x = r.origin + t * r.direction;

		// point in triangle
		float s1 = glm::length(glm::cross(x - v1, x - v2));
		float s2 = glm::length(glm::cross(x - v2, x - v3));
		float s3 = glm::length(glm::cross(x - v3, x - v1));
		float s = glm::length(glm::cross(v1 - v2, v3 - v2));
		if (s1 >= 0 && s2 >= 0 && s3 >= 0 && s1 <= s && s2 <= s && s3 <= s && abs(s1 + s2 + s3 - s) <= 0.0001f) {
			intersection = x;
			normal = glm::normalize(s1/s * n3 + s2/s * n1 + s3/s * n2);
			uv = s1/s * t3 + s2/s * t1 + s3/s * t2;

			return glm::length(r.origin - x);
		}
		else {
			return -1;
		}
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
__host__ __device__ void getRandomPointNormalUVOnCube(staticGeom cube, float randomSeed, glm::vec3& point, glm::vec3& normal, glm::vec2& uv){

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

	normal = getNormalOfPointOnUnitCube(point);
	uv = getUVOfPointOnUnitCube(point);

	glm::vec3 extendedPoint = point + normal;

	point = multiplyMV(cube.transform, glm::vec4(point,1.0f));
	extendedPoint = multiplyMV(cube.transform, glm::vec4(extendedPoint,1.0f));

	normal = glm::normalize(extendedPoint - point);
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
// Marsaglia (1972)
// http://mathworld.wolfram.com/SpherePointPicking.html
__host__ __device__ void getRandomPointNormalUVOnSphere(staticGeom sphere, float randomSeed, glm::vec3& point, glm::vec3& normal, glm::vec2& uv){

	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> uniformDistribution(0,1);
	float x1, x2;
	x1 = (float) uniformDistribution(rng);
	x2 = (float) uniformDistribution(rng);

	float theta = 2.0f * PI * x1;
	float phi = acos(2.0f * x2 - 1);

	float radius = 0.5f;

	glm::vec3 localPoint = glm::vec3( radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(phi));

	point = multiplyMV(sphere.transform, glm::vec4(localPoint,1.0f));
	glm::vec3 centerViewSpace = multiplyMV(sphere.transform, glm::vec4(0.0f,0.0f,0.0f,1.0f));

	normal = glm::normalize(point - centerViewSpace);
	uv = glm::vec2(0.5 + atan2(localPoint.z, localPoint.x) / (2.0f * PI), 0.5 - asin(localPoint.y) / PI);
}


//finds the axis aligned bounding box given a geomtry
__host__ __device__ void buildAABB(staticGeom& geom) {

	//build unit cube
	glm::vec3 unitXYZmin (-0.5f);
	glm::vec3 unitXYZmax (0.5f);

	//apply transforms to unit cube
	unitXYZmin = multiplyMV(geom.transform, glm::vec4(unitXYZmin, 1.0f));
	unitXYZmax = multiplyMV(geom.transform, glm::vec4(unitXYZmax, 1.0f));

	//geom.materialid==
	//take min and max of points as bounds of box
	geom.boundingBox.xyzMin = glm::min(unitXYZmin, unitXYZmax);
	geom.boundingBox.xyzMax = glm::max(unitXYZmin, unitXYZmax);
	geom.boundingBox.dimension = glm::abs(geom.boundingBox.xyzMax - geom.boundingBox.xyzMin);

}


__host__ __device__ glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale){
  glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
  glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1,0,0));
  rotationMat = rotationMat*glm::rotate(glm::mat4(), rotation.y, glm::vec3(0,1,0));
  rotationMat = rotationMat*glm::rotate(glm::mat4(), rotation.z, glm::vec3(0,0,1));
  glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
  return translationMat*rotationMat*scaleMat;
}

__host__ __device__ cudaMat4 glmMat4ToCudaMat4(glm::mat4 a){
    cudaMat4 m; a = glm::transpose(a);
    m.x = a[0];
    m.y = a[1];
    m.z = a[2];
    m.w = a[3];
    return m;
}