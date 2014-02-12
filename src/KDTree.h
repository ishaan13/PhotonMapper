#include "sceneStructs.h"
#include "KDTreeStructs.h"
//#include "intersections.h"
#include <vector>

#define SPLIT_BINS 10
#define MAX_PRIMS_PER_LEAF 5
#define MAX_TREE_DEPTH 25

enum {X_AXIS, Y_AXIS, Z_AXIS};
enum {LEFT, RIGHT, BOTTOM, TOP, BACK, FRONT};

class prim
{
public:
	triangle t;
	int index;
};

class Plane
{
public:
	int axis;
	float splitPoint;
	bool isFirst(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);
	bool isSecond(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);

	//for checking if a point is in left or right of split
	bool isPointInFirst(glm::vec3 p);
};

class KDNode
{
public:
	glm::vec3 llb; // lower left back
	glm::vec3 urf; // upper right front
	int numberOfPrims;
	int *primIndices;

	int kdIndex;
	KDNode *first;
	KDNode *second;

	Plane splitPlane;

	// ropes
	KDNode* ropes[6];

	bool isLeaf();
};

class KDTree
{
public:
	glm::vec3* vertices;
	triangle* faces;
	int numberOfVertices;
	int numberOfFaces;

	KDNode* tree;
	KDNodeGPU *gpuTree;
	int rootIndex;

	std::vector<int> primIndex;

	~KDTree();

	void buildKD(glm::vec3* vertices1, triangle* faces1, int numberOfVertices1, int numberOfFaces1);
	KDNode* buildTree(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList, int depth);

	//helper functions for building kd tree
	void splitPrimList(Plane p, std::vector<prim> primsList, std::vector<prim> &firstTempList,std::vector<prim> &secondTempList);
	Plane findSplitPlane(glm::vec3 llb, glm::vec3 urf);
	Plane findOptimalSplitPlane(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList);

	KDNode* optimize(KDNode* rope, int side, glm::vec3 llb, glm::vec3 urf);
	void processNode(KDNode* node, KDNode* ropes[]);

	float traverse(ray& r);

	//leave intersection tests here for now
	
	//for testing intersection with bounding box of node
	bool aabbIntersectionTest(glm::vec3 high, glm::vec3 low, ray& r, float& tNear, float& tFar);
	
	//checking intersection with primitives and if intersection point is between entry and exit
	float triangleIntersectionTest(ray& r, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);

	//finds which neighbor node to use from an exit point
	KDNode* findNeighbor (glm::vec3 p, KDNode* k);

	// Set indices in dfs order
	void setIndices(KDNode * current, int &index);

	// Put data into GPU friendly tree
	void setGPUTreeData(KDNode * current, KDNodeGPU *gpuTree, std::vector<int> &primIndexList);

	// wrapper to build linear node structure
	int buildGPUKDTree();
};

__device__ float traverse(ray &r, KDNodeGPU *nodes, int entryIndex, 
							glm::vec3 * cudaVertices, glm::vec3 * cudaNormals, glm::vec2 * cudaUV, triangle *cudaFaces, int * kdFaceIndexList, staticGeom * geoms,
							glm::vec3 &minIntersectionPoint, glm::vec3 &minNormal, int &intersectedGeom, int &intersectedMaterial, glm::vec2 &minUV, float &kdHeat);
