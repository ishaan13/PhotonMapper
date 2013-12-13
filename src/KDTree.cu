#include "KDTree.h"
#include "intersections.h"

#define EPSILON 0.0001f

__device__ float flt_max = FLT_MAX;

__device__ bool aabbIntersectionTestGPU(glm::vec3 low, glm::vec3 high, ray r, float &tNear, float &tFar)
{

	tNear = -FLT_MAX;
	tFar = FLT_MAX;
	float t1, t2 = 0.0f;

	//x planes
	//check if parallel to x plane
	if (abs(r.direction.x) < EPSILON) {
		if (r.origin.x < low.x || r.origin.x > high.x) {
			return false;
		}
	}
	else {
		t1 = (low.x - r.origin.x)/r.direction.x;
		t2 = (high.x - r.origin.x)/r.direction.x;
		if (t1 > t2) {                
			float temp = t2;
			t2 = t1;
			t1 = temp;
		}

		if (t1 > tNear) {
			tNear = t1;
		}

		if (t2 < tFar) {
			tFar = t2;
		} 

		if (tNear > tFar || tFar < 0) {
			return false;
		}
	}

	//y planes
	//check if parallel to y slabs
	if (abs(r.direction.y) < EPSILON) {
		//if within slabs
		if (r.origin.y < low.y || r.origin.y > high.y) {
			return false;
		}
	}
	else {
		t1 = (low.y - r.origin.y)/r.direction.y;
		t2 = (high.y - r.origin.y)/r.direction.y;
		if (t1 > t2) {                
			float temp = t2;
			t2 = t1;
			t1 = temp;
		}

		if (t1 > tNear) {
			tNear = t1;
		}

		if (t2 < tFar) {
			tFar = t2;
		} 

		if (tNear > tFar || tFar < 0) {
			return false;
		}
	}

	//z planes
	//check if parallel to z planes
	if (abs(r.direction.z) < EPSILON) {
		//if within slabs
		if (r.origin.z < low.z || r.origin.z > high.z) {
			return false;
		}
	}
	else {
		t1 = (low.z - r.origin.z)/r.direction.z;
		t2 = (high.z - r.origin.z)/r.direction.z;
		if (t1 > t2) {                
			float temp = t2;
			t2 = t1;
			t1 = temp;
		}

		if (t1 > tNear) {
			tNear = t1;
		}

		if (t2 < tFar) {
			tFar = t2;
		} 

		if (tNear > tFar || tFar < 0) {
			return false;
		}
	}

	//to take care of the case of ray bouncing in scene
	tNear = glm::max(tNear, 0.0f);

	return true;

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
		return -1;
	}
}

__device__ float traverse(ray &r, KDNodeGPU *nodes, int entryIndex, 
							glm::vec3 * cudaVertices, glm::vec3 * cudaNormals, glm::vec2 * cudaUV, triangle *cudaFaces, int * kdFaceIndexList, staticGeom * geoms,
							glm::vec3 &minIntersectionPoint, glm::vec3 &minNormal, int &intersectedGeom, int &intersectedMaterial, glm::vec2 &minUV, float &kdHeat)
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
	float prevEntry = -FLT_MAX;

	while(entry < exit && entry > prevEntry)
	{
		kdHeat += 1.0f/MAX_TREE_DEPTH;
		prevEntry = entry;
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
		aabbIntersectionTestGPU(node.llb, node.urf, r, entry, exit);
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

			// update exit point if new intersection point is closer to us
			if(intersect >= entry && intersect <= exit)
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

		glm::vec3 newEntry = r.origin + entry * r.direction;
		currentIndex = findNeighbor(newEntry, node);

		if(currentIndex == -1)
			return -1;
		else
			node = nodes[currentIndex];
	}
}