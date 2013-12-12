#include "KDTree.h"

// Helper funcitons
Plane findSplitPlane(glm::vec3 llb, glm::vec3 urf)
{
	Plane p;
	glm::vec3 diff = urf-llb;
	
	// find longest axis and split down the middle
	if(diff.x > diff.y && diff.x > diff.z)
	{
		p.axis = X_AXIS;
		p.splitPoint = llb.x + diff.x/2.0f;
	}
	else if(diff.y > diff.x && diff.y > diff.z)
	{
		p.axis = Y_AXIS;
		p.splitPoint = llb.y + diff.y/2.0f;
	}
	else
	{
		p.axis = Z_AXIS;
		p.splitPoint = llb.z + diff.z/2.0f;
	}
	return p;
}

void calculateBoundingBoxes(glm::vec3 llb, glm::vec3 urf, Plane splitPlane, glm::vec3 &firstllb, glm::vec3 &firsturf,
														glm::vec3 &secondllb, glm::vec3 &secondurf)
{
	if(splitPlane.axis == X_AXIS)
	{
		firstllb = llb;
		firsturf = glm::vec3(splitPlane.splitPoint,urf.y,urf.z);

		secondllb = glm::vec3(splitPlane.splitPoint,llb.y,llb.z);
		secondurf = urf;
	}
	else if(splitPlane.axis == Y_AXIS)
	{
		firstllb = llb;
		firsturf = glm::vec3(urf.x,splitPlane.splitPoint,urf.z);

		secondllb = glm::vec3(llb.x,splitPlane.splitPoint,llb.z);
		secondurf = urf;
	}
	else
	{
		firstllb = llb;
		firsturf = glm::vec3(urf.x,urf.y,splitPlane.splitPoint);

		secondllb = glm::vec3(llb.x,llb.y,splitPlane.splitPoint);
		secondurf = urf;
	}
}

// Plane direction checker
bool Plane::isFirst(prim p)
{
	if(axis == X_AXIS)
	{
		return ((vertices[p.t.v1].x <= splitPoint) || (vertices[p.t.v2].x <= splitPoint) || (vertices[p.t.v3].x <= splitPoint));
	}
	else if(axis == Y_AXIS)
	{
		return ((vertices[p.t.v1].y <= splitPoint) || (vertices[p.t.v2].y <= splitPoint) || (vertices[p.t.v3].y <= splitPoint));
	}
	else
	{
		return ((vertices[p.t.v1].z <= splitPoint) || (vertices[p.t.v2].z <= splitPoint) || (vertices[p.t.v3].z <= splitPoint));
	}
}

bool Plane::isSecond(prim p)
{
	if(axis == X_AXIS)
	{
		return ((vertices[p.t.v1].x >= splitPoint) || (vertices[p.t.v2].x >= splitPoint) || (vertices[p.t.v3].x >= splitPoint));
	}
	else if(axis == Y_AXIS)
	{
		return ((vertices[p.t.v1].y >= splitPoint) || (vertices[p.t.v2].y >= splitPoint) || (vertices[p.t.v3].y >= splitPoint));
	}
	else
	{
		return ((vertices[p.t.v1].z >= splitPoint) || (vertices[p.t.v2].z >= splitPoint) || (vertices[p.t.v3].z >= splitPoint));
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
bool KDTree::aabbIntersectionTest(glm::vec3 high, glm::vec3 low, ray& r, float& tNear, float& tFar) {
	
	tNear = FLT_MIN;
	tFar = FLT_MAX;
	float t1, t2 = 0.0f;

	//x planes
	//check if parallel to x plane
	if (abs(r.direction.x) < FLT_EPSILON) {
		if (r.direction.x < low.x || r.direction.x > high.x) {
			return false;
		}
	}
	else {
		t1 = (low.x - r.origin.x)/r.direction.x;
		t2 = (high.x - r.origin.x)/r.direction.x;
		if (t1 > t2) {		
			float temp = t1;
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
	if (abs(r.direction.y) < FLT_EPSILON) {
		//if within slabs
		if (r.direction.y < low.y || r.direction.y > high.y) {
			return false;
		}
	}
	else {
		t1 = (low.y - r.origin.y)/r.direction.y;
		t2 = (high.y - r.origin.y)/r.direction.y;
		if (t1 > t2) {		
			float temp = t1;
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
	if (abs(r.direction.z) < FLT_EPSILON) {
		//if within slabs
		if (r.direction.z < low.z || r.direction.z > high.z) {
			return false;
		}
	}
	else {
		t1 = (low.z - r.origin.z)/r.direction.z;
		t2 = (high.z - r.origin.z)/r.direction.z;
		if (t1 > t2) {		
			float temp = t1;
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

// Wrapepr function which'll do magic
void KDTree::buildKD()
{
	// find two corners of bounding box of scene
	glm::vec3 llb = glm::vec3(FLT_MAX);
	glm::vec3 urf = glm::vec3(-FLT_MAX); // FLT_MIN is the smallest +ve number representable
	for(int i=0; i<numberOfFaces; i++)
	{
		// Update Min
		if(vertices[faces[i].v1].x < llb.x)	llb.x = vertices[faces[i].v1].x;
		if(vertices[faces[i].v2].x < llb.x)	llb.x = vertices[faces[i].v2].x;
		if(vertices[faces[i].v3].x < llb.x)	llb.x = vertices[faces[i].v3].x;

		if(vertices[faces[i].v1].y < llb.y)	llb.y = vertices[faces[i].v1].y;
		if(vertices[faces[i].v2].y < llb.y)	llb.y = vertices[faces[i].v2].y;
		if(vertices[faces[i].v3].y < llb.y)	llb.y = vertices[faces[i].v3].y;

		if(vertices[faces[i].v1].z < llb.z)	llb.z = vertices[faces[i].v1].z;
		if(vertices[faces[i].v2].z < llb.z)	llb.z = vertices[faces[i].v2].z;
		if(vertices[faces[i].v3].z < llb.z)	llb.z = vertices[faces[i].v3].z;

		//faster way?
		//llb = glm::min(vertices[faces[i].v1], vertices[faces[i].v2]);
		//llb = glm::min(llb, vertices[faces[i].v3]);

		// Update Max
		if(vertices[faces[i].v1].x > urf.x)	urf.x = vertices[faces[i].v1].x;
		if(vertices[faces[i].v2].x > urf.x)	urf.x = vertices[faces[i].v2].x;
		if(vertices[faces[i].v3].x > urf.x)	urf.x = vertices[faces[i].v3].x;

		if(vertices[faces[i].v1].y > urf.y)	urf.y = vertices[faces[i].v1].y;
		if(vertices[faces[i].v2].y > urf.y)	urf.y = vertices[faces[i].v2].y;
		if(vertices[faces[i].v3].y > urf.y)	urf.y = vertices[faces[i].v3].y;

		if(vertices[faces[i].v1].z > urf.z)	urf.z = vertices[faces[i].v1].z;
		if(vertices[faces[i].v2].z > urf.z)	urf.z = vertices[faces[i].v2].z;
		if(vertices[faces[i].v3].z > urf.z)	urf.z = vertices[faces[i].v3].z;

		//urf = glm::max(vertices[faces[i].v1], vertices[faces[i].v2]);
		//urf = glm::max(urf, vertices[faces[i].v3]);
	
	}

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
	tree = buildTree(llb,urf,primList);

	// build rope structure
	KDNode* ropes[] = {NULL, NULL, NULL, NULL, NULL, NULL};
	processNode(tree, ropes);
}

// recursive build
KDNode * KDTree::buildTree(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList)
{
	KDNode * current;

	current = new KDNode;
	current->llb = llb;
	current->urf = urf;
	current->first = NULL;
	current->second = NULL;

	// Recursion termination condition
	if(primsList.size() <= MAX_PRIMS_PER_LEAF)
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

		// Make two lists of primitives based on positive and negative side of this plane. if confused, put in both.
		std::vector<prim> firstPrimsList;
		std::vector<prim> secondPrimsList;
		for(int i=0; i< primsList.size(); i++)
		{
			if(splitPlane.isFirst(primsList[i]))
			{
				firstPrimsList.push_back(primsList[i]);
			}
			else if(splitPlane.isSecond(primsList[i]))
			{
				secondPrimsList.push_back(primsList[i]);
			}
		}

		glm::vec3 firstllb, firsturf;
		glm::vec3 secondllb, secondurf;

		calculateBoundingBoxes(llb,urf,splitPlane,firstllb,firsturf,secondllb,secondurf);

		// Split the primitives based on left and right
		current->splitPlane = splitPlane;
		current->first = buildTree(firstllb,firsturf,firstPrimsList);
		current->second = buildTree(secondllb, secondurf,secondPrimsList);
		current->numberOfPrims = 0;
		current->primIndices = NULL;
	}
	return current;
}

// optimize rope
KDNode* KDTree::optimize(KDNode* rope, int side, glm::vec3 llb, glm::vec3 urf)
{
	Plane splitPlane = rope->splitPlane;

	while (!rope->isLeaf())
	{
		bool splitted = false;

		if (side == LEFT || side == RIGHT) {
			if (splitPlane.axis == X_AXIS) {
				rope = side == RIGHT ? rope->first : rope->second;
				splitted = true;
			}
			else if (splitPlane.axis == Y_AXIS) {
				if (splitPlane.splitPoint >= urf.y) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint <= llb.y) {
					rope = rope->second;
					splitted = true;
				}
			}
			else {
				if (splitPlane.splitPoint >= urf.z) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint <= llb.z) {
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
				if (splitPlane.splitPoint >= urf.x) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint <= llb.x) {
					rope = rope->second;
					splitted = true;
				}
			}
			else {
				if (splitPlane.splitPoint >= urf.z) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint <= llb.z) {
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
				if (splitPlane.splitPoint >= urf.x) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint <= llb.x) {
					rope = rope->second;
					splitted = true;
				}
			}
			else {
				if (splitPlane.splitPoint >= urf.y) {
					rope = rope->first;
					splitted = true;
				}
				else if (splitPlane.splitPoint <= llb.y) {
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
			if (node->ropes[i] != NULL) {
				node->ropes[i] = optimize(node->ropes[i], i, node->llb, node->urf);
			}
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

float KDTree::traverse(ray& r, KDNode* node, std::vector<prim> primsList) {
	
	float entry = FLT_MIN; 
	float exit = FLT_MAX;

	//find entry and exit distances of the ray into the tree
	//if (!aabbIntersectionTest(node->urf, node->llb, r, entry, exit)) {
		//no intersection, so return
		//return -1;
	//}

	while (entry < exit) {
		
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

		//now at a leaf, check for intersection with primitives
		for (int i = 0; i < node->numberOfPrims; ++i) {

			//intersect with triangle
			prim tri = primsList[node->primIndices[i]];
		
		}	//exit leaf node

		entry = exit;
		//go to the next exit node
		
	}

}




