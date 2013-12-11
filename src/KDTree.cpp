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

}

// recursive build
KDNode * buildTree(glm::vec3 llb, glm::vec3 urf, std::vector<prim> primsList)
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
		current->first = buildTree(firstllb,firsturf,firstPrimsList);
		current->second = buildTree(secondllb, secondurf,secondPrimsList);
		current->numberOfPrims = 0;
		current->primIndices = NULL;
	}
	return current;
}

