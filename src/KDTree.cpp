#include "KDTree.h"
#define EPSILON 0.00001

// Helper funcitons

float surfaceArea(glm::vec3 llb, glm::vec3 urf)
{
	glm::vec3 diff = urf - llb;
	return 2.0f*(diff.x*diff.y + diff.y*diff.z + diff.z*diff.x);
}

void calculateBoundingBoxes(glm::vec3 llb, glm::vec3 urf, Plane splitPlane, glm::vec3 &firstllb, glm::vec3 &firsturf,
	glm::vec3 &secondllb, glm::vec3 &secondurf)
{
	//glm::vec3 epsilonVec(0.0001f);

	////adding epsilon checks
	//if(splitPlane.axis == X_AXIS)
	//{
	//        firstllb = llb - epsilonVec;
	//        firsturf = glm::vec3(splitPlane.splitPoint, urf.y, urf.z) + epsilonVec;

	//        secondllb = glm::vec3(splitPlane.splitPoint, llb.y, llb.z) - epsilonVec;
	//        secondurf = urf + epsilonVec;
	//}
	//else if(splitPlane.axis == Y_AXIS)
	//{
	//        firstllb = llb - epsilonVec;
	//        firsturf = glm::vec3(urf.x,splitPlane.splitPoint, urf.z) + epsilonVec;

	//        secondllb = glm::vec3(llb.x,splitPlane.splitPoint,llb.z) - epsilonVec;
	//        secondurf = urf + epsilonVec;
	//}
	//else
	//{
	//        firstllb = llb - epsilonVec;
	//        firsturf = glm::vec3(urf.x,urf.y,splitPlane.splitPoint) + epsilonVec;

	//        secondllb = glm::vec3(llb.x,llb.y,splitPlane.splitPoint) - epsilonVec;
	//        secondurf = urf + epsilonVec;
	//}

	//adding epsilon checks
	if(splitPlane.axis == X_AXIS)
	{
		firstllb = llb - glm::vec3(EPSILON, 0.0f, 0.0f);
		firsturf = glm::vec3(splitPlane.splitPoint + EPSILON,urf.y,urf.z);

		secondllb = glm::vec3(splitPlane.splitPoint - EPSILON,llb.y,llb.z);
		secondurf = urf + glm::vec3(EPSILON, 0.0f, 0.0f);
	}
	else if(splitPlane.axis == Y_AXIS)
	{
		firstllb = llb - glm::vec3(0.0f, EPSILON, 0.0f);
		firsturf = glm::vec3(urf.x,splitPlane.splitPoint + EPSILON,urf.z);

		secondllb = glm::vec3(llb.x,splitPlane.splitPoint - EPSILON,llb.z);
		secondurf = urf + glm::vec3(0.0f, EPSILON, 0.0f);
	}
	else
	{
		firstllb = llb - glm::vec3(0.0f, 0.0f, EPSILON);
		firsturf = glm::vec3(urf.x,urf.y,splitPlane.splitPoint + EPSILON);

		secondllb = glm::vec3(llb.x,llb.y,splitPlane.splitPoint - EPSILON);
		secondurf = urf + glm::vec3(0.0f, 0.0, EPSILON);
	}

	//if(splitPlane.axis == X_AXIS)
	//{
	//        firstllb = llb;
	//        firsturf = glm::vec3(splitPlane.splitPoint,urf.y,urf.z);

	//        secondllb = glm::vec3(splitPlane.splitPoint,llb.y,llb.z);
	//        secondurf = urf;
	//}
	//else if(splitPlane.axis == Y_AXIS)
	//{
	//        firstllb = llb;
	//        firsturf = glm::vec3(urf.x,splitPlane.splitPoint,urf.z);

	//        secondllb = glm::vec3(llb.x,splitPlane.splitPoint,llb.z);
	//        secondurf = urf;
	//}
	//else
	//{
	//        firstllb = llb;
	//        firsturf = glm::vec3(urf.x,urf.y,splitPlane.splitPoint);

	//        secondllb = glm::vec3(llb.x,llb.y,splitPlane.splitPoint);
	//        secondurf = urf;
	//}

}


// Plane direction checker
bool Plane::isFirst(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
{
	if(axis == X_AXIS)
	{
		return ((v1.x <= splitPoint) || (v2.x <= splitPoint) || (v3.x <= splitPoint));
	}
	else if(axis == Y_AXIS)
	{
		return ((v1.y <= splitPoint) || (v2.y <= splitPoint) || (v3.y <= splitPoint));
	}
	else
	{
		return ((v1.z <= splitPoint) || (v2.z <= splitPoint) || (v3.z <= splitPoint));
	}
}

bool Plane::isSecond(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
{
	if(axis == X_AXIS)
	{
		return ((v1.x >= splitPoint) || (v2.x >= splitPoint) || (v3.x >= splitPoint));
	}
	else if(axis == Y_AXIS)
	{
		return ((v1.y >= splitPoint) || (v2.y >= splitPoint) || (v3.y >= splitPoint));
	}
	else
	{
		return ((v1.z >= splitPoint) || (v2.z >= splitPoint) || (v3.z >= splitPoint));
	}
}

//return true if point is in left split, false if in right
bool Plane::isPointInFirst(glm::vec3 p) {

	if (axis == X_AXIS) {
		return p.x <= splitPoint;        
	}

	else if (axis == Y_AXIS) {
		return p.y <= splitPoint;
	}

	else {
		return p.z <= splitPoint;
	}
}

bool KDNode::isLeaf()
{
	return (first == NULL && second == NULL);
}

//box intersection test with bounding boxes
bool KDTree::aabbIntersectionTest(glm::vec3 low, glm::vec3 high, ray& r, float& tNear, float& tFar) {

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

	return true;

}

float KDTree::triangleIntersectionTest(ray& r, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {
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
	float s1 = glm::dot(glm::cross(v2 - v1, x - v1), n);
	float s2 = glm::dot(glm::cross(v3 - v2, x - v2), n);
	float s3 = glm::dot(glm::cross(v1 - v3, x - v3), n);

	if (s1 >= 0 && s2 >= 0 && s3 >= 0) {
		return glm::length(r.origin - x);
	}
	else {
		return -1;
	}
}

//given a point, find which face the point is on and use that to find the neighbor using ropes
// Make sure that the node that the rope takes us to has an exit point larger than the old entry
KDNode* KDTree:: findNeighbor (glm::vec3 p, KDNode* k) {

	if (abs(p.x - k->llb.x) < EPSILON) {
		return k->ropes[LEFT];                
	}
	else if (abs(p.x - k->urf.x) < EPSILON) {
		return k->ropes[RIGHT];
	}
	else if (abs(p.y - k->llb.y) < EPSILON) {
		return k->ropes[BOTTOM];
	}
	else if (abs(p.y - k->urf.y) < EPSILON) {
		return k->ropes[TOP];
	}
	else if (abs(p.z - k->llb.z) < EPSILON) {
		return k->ropes[BACK];
	}
	else if (abs(p.z - k->urf.z) < EPSILON) {
		return k->ropes[FRONT];
	}
	else {
		return NULL;
	}

}


// Wrapepr function which'll do magic
void KDTree::buildKD(glm::vec3* vertices1, triangle* faces1, int numberOfVertices1, int numberOfFaces1)
{
	vertices = vertices1;
	faces = faces1;
	numberOfVertices = numberOfVertices1;
	numberOfFaces = numberOfFaces1;

	// find two corners of bounding box of scene
	glm::vec3 llb = glm::vec3(FLT_MAX);
	glm::vec3 urf = glm::vec3(-FLT_MAX); // FLT_MIN is the smallest +ve number representable
	for(int i=0; i<numberOfFaces; i++)
	{
		// Update Min
		if(vertices[faces[i].v1].x < llb.x)        llb.x = vertices[faces[i].v1].x;
		if(vertices[faces[i].v2].x < llb.x)        llb.x = vertices[faces[i].v2].x;
		if(vertices[faces[i].v3].x < llb.x)        llb.x = vertices[faces[i].v3].x;

		if(vertices[faces[i].v1].y < llb.y)        llb.y = vertices[faces[i].v1].y;
		if(vertices[faces[i].v2].y < llb.y)        llb.y = vertices[faces[i].v2].y;
		if(vertices[faces[i].v3].y < llb.y)        llb.y = vertices[faces[i].v3].y;

		if(vertices[faces[i].v1].z < llb.z)        llb.z = vertices[faces[i].v1].z;
		if(vertices[faces[i].v2].z < llb.z)        llb.z = vertices[faces[i].v2].z;
		if(vertices[faces[i].v3].z < llb.z)        llb.z = vertices[faces[i].v3].z;

		//faster way?
		//llb = glm::min(vertices[faces[i].v1], vertices[faces[i].v2]);
		//llb = glm::min(llb, vertices[faces[i].v3]);

		// Update Max
		if(vertices[faces[i].v1].x > urf.x)        urf.x = vertices[faces[i].v1].x;
		if(vertices[faces[i].v2].x > urf.x)        urf.x = vertices[faces[i].v2].x;
		if(vertices[faces[i].v3].x > urf.x)        urf.x = vertices[faces[i].v3].x;

		if(vertices[faces[i].v1].y > urf.y)        urf.y = vertices[faces[i].v1].y;
		if(vertices[faces[i].v2].y > urf.y)        urf.y = vertices[faces[i].v2].y;
		if(vertices[faces[i].v3].y > urf.y)        urf.y = vertices[faces[i].v3].y;

		if(vertices[faces[i].v1].z > urf.z)        urf.z = vertices[faces[i].v1].z;
		if(vertices[faces[i].v2].z > urf.z)        urf.z = vertices[faces[i].v2].z;
		if(vertices[faces[i].v3].z > urf.z)        urf.z = vertices[faces[i].v3].z;

		//urf = glm::max(vertices[faces[i].v1], vertices[faces[i].v2]);
		//urf = glm::max(urf, vertices[faces[i].v3]);

	}

	//add epsilon values for LLB and URF
	llb += glm::vec3(-EPSILON); 
	urf += glm::vec3(EPSILON);

	// build vector of triangles
	std::vector<prim> primList;
	for(int i=0; i<numberOfFaces; i++)
	{
		prim p;
		p.index = i;
		p.t = faces[i];
		primList.push_back(p);
	}
	// call recursive build

	tree = buildTree(llb,urf,primList, 0);
	// build rope structure
	KDNode* ropes[] = {NULL, NULL, NULL, NULL, NULL, NULL};
	processNode(tree, ropes);
}

// recursive build
KDNode * KDTree::buildTree(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList, int depth)
{
	KDNode * current;

	current = new KDNode;
	current->llb = llb;
	current->urf = urf;
	current->first = NULL;
	current->second = NULL;

	// Recursion termination condition
	if(primsList.size() <= MAX_PRIMS_PER_LEAF || depth >= MAX_TREE_DEPTH)
	{
		// if primList is empty, this volume is empty; return with bounding box
		if(primsList.size() == 0)
		{
			current->numberOfPrims = 0;
			current->primIndices = NULL;
		}
		else
		{
			current->numberOfPrims = primsList.size();
			current->primIndices = new int[current->numberOfPrims];
			for(int i=0; i<primsList.size(); i++)
			{
				// store the index in the original triangle list
				current->primIndices[i] = primsList[i].index;
			}
		}
	}
	else
	{

		// Split the tree
		Plane splitPlane = findSplitPlane(llb,urf);

		//optimal split
		//Plane splitPlane = findOptimalSplitPlane(llb, urf, primsList);

		// Make two lists of primitives based on positive and negative side of this plane. if confused, put in both.
		std::vector<prim> firstPrimsList;
		std::vector<prim> secondPrimsList;
		for(int i=0; i< primsList.size(); i++)
		{
			glm::vec3 v1 = vertices[faces[primsList[i].index].v1];
			glm::vec3 v2 = vertices[faces[primsList[i].index].v2];
			glm::vec3 v3 = vertices[faces[primsList[i].index].v3];

			if(splitPlane.isFirst(v1, v2, v3))
			{
				firstPrimsList.push_back(primsList[i]);
			}
			if(splitPlane.isSecond(v1, v2, v3))
			{
				secondPrimsList.push_back(primsList[i]);
			}
		}

		glm::vec3 firstllb, firsturf;
		glm::vec3 secondllb, secondurf;

		calculateBoundingBoxes(llb,urf,splitPlane,firstllb,firsturf,secondllb,secondurf);

		// Split the primitives based on left and right
		current->splitPlane = splitPlane;
		current->first = buildTree(firstllb,firsturf,firstPrimsList, depth+1);
		current->second = buildTree(secondllb, secondurf,secondPrimsList, depth+1);
		current->numberOfPrims = 0;
		current->primIndices = NULL;
	}
	return current;
}

//helper functions
void KDTree::splitPrimList(Plane p, std::vector<prim> primsList, std::vector<prim> &firstTempList,std::vector<prim> &secondTempList)
{
	for(int i=0; i< primsList.size(); i++)
	{
		glm::vec3 v1 = vertices[faces[primsList[i].index].v1];
		glm::vec3 v2 = vertices[faces[primsList[i].index].v2];
		glm::vec3 v3 = vertices[faces[primsList[i].index].v3];

		if(p.isFirst(v1, v2, v3))
		{
			firstTempList.push_back(primsList[i]);
		}
		if(p.isSecond(v1, v2, v3))
		{
			secondTempList.push_back(primsList[i]);
		}
	}
}

Plane KDTree::findSplitPlane(glm::vec3 llb, glm::vec3 urf)
{
	Plane p;
	glm::vec3 diff = urf-llb;

	//if (abs(urf.y-6.0f) < 0.00001f && abs(llb.y-2.0f) < 0.00001f) {
	//        int debug = 1;
	//}

	// find longest axis and split down the middle
	if(diff.x >= diff.y && diff.x >= diff.z)
	{
		p.axis = X_AXIS;
		p.splitPoint = llb.x + diff.x/2.0f;
	}
	else if(diff.y >= diff.x && diff.y >= diff.z)
	{
		p.axis = Y_AXIS;
		p.splitPoint = llb.y + diff.y/2.0f;
	}
	else if(diff.z >= diff.y && diff.z >= diff.x)
	{
		p.axis = Z_AXIS;
		p.splitPoint = llb.z + diff.z/2.0f;
	}
	return p;
}

Plane KDTree::findOptimalSplitPlane(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList)
{
	Plane p;
	glm::vec3 diff = urf-llb;

	// find longest axis and split down the middle
	if(diff.x >= diff.y && diff.x >= diff.z)
	{
		p.axis = X_AXIS;
		//p.splitPoint = llb.x + diff.x/2.0f;
	}
	else if(diff.y >= diff.x && diff.y >= diff.z)
	{
		p.axis = Y_AXIS;
		//p.splitPoint = llb.y + diff.y/2.0f;
	}
	else
	{
		p.axis = Z_AXIS;
		//p.splitPoint = llb.z + diff.z/2.0f;
	}

	float minCost = FLT_MAX;
	Plane optimalPlane;
	float cost;

	std::vector<prim> firstTempList;
	std::vector<prim> secondTempList;

	for(int i=0; i < SPLIT_BINS-1; i++)
	{

		firstTempList.resize(0);
		secondTempList.resize(0);

		if(p.axis == X_AXIS)
			p.splitPoint = llb.x + (i+1) * diff.x / (SPLIT_BINS) ;
		else if(p.axis == Y_AXIS)
			p.splitPoint = llb.y + (i+1) * diff.y / (SPLIT_BINS) ;
		else
			p.splitPoint = llb.z + (i+1) * diff.z / (SPLIT_BINS) ;

		splitPrimList(p,primsList,firstTempList,secondTempList);
		glm::vec3 fllb, furf;
		glm::vec3 sllb, surf;
		calculateBoundingBoxes(llb,urf,p,fllb,furf,sllb,surf);
		cost = surfaceArea(fllb,furf)*firstTempList.size() + surfaceArea(sllb,surf)*secondTempList.size();
		if(cost < minCost)
		{
			minCost = cost;
			optimalPlane = p;
		}
	}
	return optimalPlane;
}

// optimize rope
KDNode* KDTree::optimize(KDNode* rope, int side, glm::vec3 llb, glm::vec3 urf)
{
	while (!rope->isLeaf())
	{
		bool splitted = false;

		Plane splitPlane = rope->splitPlane;
		if (side == LEFT || side == RIGHT) {
			if (splitPlane.axis == X_AXIS) {
				rope = side == RIGHT ? rope->first : rope->second;
				splitted = true;
			}
			else if (splitPlane.axis == Y_AXIS) {
				if (splitPlane.splitPoint > urf.y + 0.0001) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint < llb.y - 0.0001) {
					rope = rope->second;
					splitted = true;
				}
			}
			else {
				if (splitPlane.splitPoint > urf.z + 0.0001) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint < llb.z -  0.0001) {
					rope = rope->second;
					splitted = true;
				}
			}
		}
		else if (side == TOP || side == BOTTOM) {
			if (splitPlane.axis == Y_AXIS) {
				rope = side == TOP ? rope->first : rope->second;
				splitted = true;
			}
			else if (splitPlane.axis == X_AXIS) {
				if (splitPlane.splitPoint > urf.x + 0.0001) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint < llb.x - 0.0001) {
					rope = rope->second;
					splitted = true;
				}
			}
			else {
				if (splitPlane.splitPoint > urf.z + 0.0001) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint < llb.z - 0.0001) {
					rope = rope->second;
					splitted = true;
				}
			}
		}
		else {
			if (splitPlane.axis == Z_AXIS) {
				rope = side == FRONT ? rope->first : rope->second;
				splitted = true;
			}
			else if (splitPlane.axis == X_AXIS) {
				if (splitPlane.splitPoint > urf.x + 0.0001) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint < llb.x - 0.0001) {
					rope = rope->second;
					splitted = true;
				}
			}
			else {
				if (splitPlane.splitPoint > urf.y + 0.0001) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint < llb.y - 0.0001) {
					rope = rope->second;
					splitted = true;
				}
			}
		}

		if (!splitted) {
			break;
		}
	}

	return rope;
}

void KDTree::processNode(KDNode* node, KDNode* ropes[])
{
	if (node->isLeaf()) {
		for (int i=0; i<6; ++i) {
			node->ropes[i] = ropes[i];
		}
	}
	else {
		for (int i=0; i<6; ++i) {
			if (ropes[i] != NULL) {
				ropes[i] = optimize(ropes[i], i, node->llb, node->urf);
			}
		}

		int side1, side2;
		Plane splitPlane = node->splitPlane;
		if (splitPlane.axis == X_AXIS) {
			side1 = LEFT;
			side2 = RIGHT;
		}
		else if (splitPlane.axis == Y_AXIS) {
			side1 = BOTTOM;
			side2 = TOP;
		}
		else {
			side1 = BACK;
			side2 = FRONT;
		}

		KDNode* ropes1[6]; // the ropes of the node's first child
		for (int i=0; i<6; ++i) {
			ropes1[i] = ropes[i];
		}
		ropes1[side2] = node->second;
		processNode(node->first, ropes1);

		KDNode* ropes2[6]; // the ropes of the node's second child
		for (int i=0; i<6; ++i) {
			ropes2[i] = ropes[i];
		}
		ropes2[side1] = node->first;
		processNode(node->second, ropes2);
	}
}

float KDTree::traverse(ray& r) {

	float entry = -FLT_MAX; 
	float exit = FLT_MAX;

	KDNode * node = tree;

	//find entry and exit distances of the ray into the tree
	if (!aabbIntersectionTest(node->llb, node->urf, r, entry, exit)) {
		//no intersection, so return
		//std::cout<<"no intersection with KD root"<<std::endl;
		return -1;
	}

	//save this value for going to neighbor nodes, since we update exit later
	float rootExit = exit;
	float prevEntry = -FLT_MAX;

	while (entry - exit < -EPSILON && entry > prevEntry) {
		
		prevEntry = entry;

		//downward traversal to find a leaf node
		glm::vec3 pEntry = r.origin + entry * r.direction;

		while (!node->isLeaf()) {
			//check which child to traverse down on
			if (node->splitPlane.isPointInFirst(pEntry)) {
				node = node->first;
			} 
			else {
				node = node->second;
			}
		}
		aabbIntersectionTest(node->llb, node->urf, r, entry, exit);
		bool intersectionFound = false;
		//now at a leaf, check for intersection with primitives
		for (int i = 0; i < node->numberOfPrims; ++i) {

			//intersect with triangle in range of entry and exit
			int pIndex = node->primIndices[i];
			triangle tri = faces[pIndex];
			glm::vec3 v1 = vertices[tri.v1];
			glm::vec3 v2 = vertices[tri.v2];
			glm::vec3 v3 = vertices[tri.v3];
			float intersect = triangleIntersectionTest(r, v1, v2, v3);        

			//std::cout<<intersect<<std::endl;

			//update exit point if new intersection point is closer to us
			if (intersect >= entry && intersect <= exit) {
				intersectionFound = true;
				exit = intersect;
			}
		}        //exit leaf node

		//if intersection found in this node, it is the closest one, so return
		if (intersectionFound) {
			return exit;
		}

		//update entry and reset exit
		entry = exit;
		exit = rootExit;

		//if no intersection, go to the next node using rope
		//if no more neigbours, return -1
		glm::vec3 newEntryPoint = r.origin + entry * r.direction;
		node = findNeighbor(newEntryPoint, node);

		//return -1 if no neighbors found
		if (!node) {
			return -1;
		}
	}

	return -1;
}

// Traverse in breadth first order and  set indices
void KDTree::setIndices(KDNode * current, int &index)
{
	if(current->isLeaf())
	{
		current->kdIndex = index;
		index++;
	}
	else
	{
		if(current->first != NULL)
		{
			setIndices(current->first, index);
		}
		if(current->second != NULL)
		{
			setIndices(current->second, index);
		}
		current->kdIndex = index;
		index++;
	}
}

void KDTree::setGPUTreeData(KDNode * current, KDNodeGPU *gpuTree, std::vector<int> &primIndexList)
{
	int index = current->kdIndex;
	gpuTree[index].llb					= current->llb;
	gpuTree[index].urf					= current->urf;
	gpuTree[index].splitPlane.axis		= current->splitPlane.axis;
	gpuTree[index].splitPlane.splitPoint = current->splitPlane.splitPoint;

	if(current->isLeaf())
	{
		gpuTree[index].first			= -1;
		gpuTree[index].second			= -1;
		gpuTree[index].numPrims			= current->numberOfPrims;
		gpuTree[index].startPrimIndex	= primIndex.size();		
		for(int i=0; i< 6; i++)
		{
			if(current->ropes[i] != NULL)
				gpuTree[index].ropes[i] = current->ropes[i]->kdIndex;
			else
				gpuTree[index].ropes[i] = -1;
		}

		for(int i=0; i< current->numberOfPrims; i++)
		{
			primIndex.push_back(current->primIndices[i]);
		}

	}
	else
	{
		if(current->first != NULL)
		{
			setGPUTreeData(current->first, gpuTree, primIndex);
			gpuTree[index].first = current->first->kdIndex;
		}
		if(current->second!= NULL)
		{
			setGPUTreeData(current->second, gpuTree, primIndex);
			gpuTree[index].second = current->second->kdIndex;
		}

		// set data
		gpuTree[index].numPrims			= 0;
		gpuTree[index].startPrimIndex	= -1;

		for(int i=0; i< 6; i++)
			gpuTree[index].ropes[i] = -1;
	}
}

// traverse and build gpu kdtree
int  KDTree::buildGPUKDTree()
{
	int numElts = 0;
	if(tree!=NULL)
	{
		setIndices(tree, numElts);

		gpuTree = new KDNodeGPU[numElts];

		setGPUTreeData(tree, gpuTree, primIndex);

		rootIndex = tree->kdIndex;

	}
	return numElts;
}