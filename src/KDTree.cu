#include "KDTree.h"
#include "intersections.h"

#define EPSILON 0.0001f

__device__ float flt_max = FLT_MAX;

__device__ int aabbIntersectionTestGPU(glm::vec3, glm::vec3, ray r, float &entry, float &exit)
{
	return -1;
}

__device__ bool isLeaf(KDNodeGPU node)
{
	return ((node.first == -1) && (node.second == -1));
}

__device__ bool isPointInFirst(PlaneGPU splitPlane, glm::vec3 p)
{
	if (splitPlane.axis == X_AXIS) {
		return p.x <= splitPlane.splitPoint;	
	}

	else if (splitPlane.axis == Y_AXIS) {
		return p.y <= splitPlane.splitPoint;
	}

	else {
		return p.z <= splitPlane.splitPoint;
	}
}

__device__ int findNeighbor(glm::vec3 p, KDNodeGPU node)
{
	if (abs(p.x - node.llb.x) < EPSILON) {
		return node.ropes[LEFT];		
	}
	else if (abs(p.x - node.urf.x) < EPSILON) {
		return node.ropes[RIGHT];
	}
	else if (abs(p.y - node.llb.y) < EPSILON) {
		return node.ropes[BOTTOM];
	}
	else if (abs(p.y - node.urf.y) < EPSILON) {
		return node.ropes[TOP];
	}
	else if (abs(p.z - node.llb.z) < EPSILON) {
		return node.ropes[BACK];
	}
	else if (abs(p.z - node.urf.z) < EPSILON) {
		return node.ropes[FRONT];
	}
	else {
		return NULL;
	}
}

__device__ float traverse(ray &r, KDNodeGPU *nodes, int entryIndex, 
							glm::vec3 * cudaVertices, glm::vec3 * cudaNormals, glm::vec2 * cudaUV, triangle *cudaFaces, int * kdFaceIndexList, staticGeom * geoms,
							glm::vec3 &minIntersectionPoint, glm::vec3 &minNormal, int &intersectedGeom, int &intersectedMaterial, glm::vec2 &minUV)
{
	float entry = -1;
	float exit = flt_max;

	int currentIndex = entryIndex;

	KDNodeGPU node = nodes[entryIndex]; 

	// Find entry and exit distances of ray into the tree;
	if (!aabbIntersectionTestGPU(node.llb, node.urf, r, entry, exit))
	{
		// No intersection
		return -1;
	}

	// Save the kd-tree exit
	float rootExit = exit;

	while(entry < exit)
	{
		// Downward traversal to find a leaf node
		glm::vec3 pEntry = r.origin + entry * r.direction;

		while(!isLeaf(node))
		{
			// Check which of the children to traverse down
			if(isPointInFirst(node.splitPlane,pEntry))
			{
				currentIndex = node.first;
			}
			else
			{
				currentIndex = node.second;
			}
			node = nodes[currentIndex];
		}

		bool intersectionFound = false;

		// Now at a leaf, check intersection with primitives in the leaf
		for(int i=node.startPrimIndex; i<node.startPrimIndex+node.numPrims; i++)
		{
			// check for intersection
			float intersect = -1;

			
			triangle t = cudaFaces[kdFaceIndexList[i]];
			//glm::vec3 v1 = t.v1

			glm::vec3 v1 = cudaVertices[t.v1];
			glm::vec3 v2 = cudaVertices[t.v2];
			glm::vec3 v3 = cudaVertices[t.v3];
			glm::vec3 n1 = cudaNormals[t.n1];
			glm::vec3 n2 = cudaNormals[t.n2];
			glm::vec3 n3 = cudaNormals[t.n3];
			glm::vec2 t1 = cudaUV[t.t1];
			glm::vec2 t2 = cudaUV[t.t2];
			glm::vec2 t3 = cudaUV[t.t3];
			glm::vec3 intersection;
			glm::vec3 normal;
			glm::vec2 uv;
			
			intersect = triangleIntersectionTest(v1, v2, v3, n1, n2, n3, t1, t2, t3, r, intersection, normal, uv);


			//float triangleIntersectionTest(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 n1, glm::vec3 n2, glm::vec3 n3,
			//								glm::vec2 t1, glm::vec2 t2, glm::vec2 t3, ray r, glm::vec3& intersection, glm::vec3& normal, glm::vec2& uv);

			// update exit point if new intersection point is closer to us
			if(intersect >= entry && intersect <exit)
			{
				intersectionFound = true;
				exit = intersect;
				minIntersectionPoint = intersection;
				minNormal = normal;
				intersectedGeom = t.geomid;
				intersectedMaterial = geoms[intersectedGeom].materialid;
				minUV = uv;
			}

		}
		// Exit leaf node

		// intersection was found in this node and it was the closest
		if(intersectionFound)
		{
			return exit;
		}

		// Update the new entry point to the exit point
		entry = exit;
		exit = rootExit;	// reset exit point to the exit point of the whole kd-tree

		// if no intersection, go to next node using ropess
		// if no neighbors, return -1;

		glm::vec3 newEntry = r.origin + exit * r.direction;
		currentIndex = findNeighbor(newEntry, node);

		if(currentIndex == -1)
			return -1;
		else
			node = nodes[currentIndex];
	}
}