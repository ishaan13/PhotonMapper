#include "sceneStructs.h"
#include <vector>

#define MAX_PRIMS_PER_LEAF 10

enum {X_AXIS, Y_AXIS, Z_AXIS};
enum {LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK};

extern glm::vec3* vertices;
extern triangle* faces;
extern int numberOfVertices;
extern int numberOfFaces;
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
	bool isFirst(prim p);
	bool isSecond(prim p);
};

class KDNode
{
public:
	glm::vec3 llb; // lower left back
	glm::vec3 urf; // upper right front
	int numberOfPrims;
	int *primIndices;

	KDNode *first; // left/bottom/back
	KDNode *second; // right/top/front

	Plane splitPlane;

	// ropes
	KDNode* ropes[6];

	bool isLeaf();
};

class KDTree
{
public:
	KDNode* tree;
	void buildKD();
	KDNode* buildTree(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList);

	KDNode* optimize(KDNode* rope, int side, glm::vec3 llb, glm::vec3 urf);
	KDNode* processNode(KDNode* node, KDNode* ropes[], glm::vec3 llb, glm::vec3 urf);
};