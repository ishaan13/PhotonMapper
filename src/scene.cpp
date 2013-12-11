// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
// Edited by Liam Boone for use with CUDA v5.5

#include <iostream>
#include "scene.h"
#include <cstring>
#include "stb_image/stb_image.h"

scene::scene(string filename){
	vertexcount = 0;
	normalcount = 0;
	uvcount = 0;
	widthcount = 0;
	maxheight = 0;
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
		while(fp_in.good()){
			string line;
			utilityCore::safeGetline(fp_in,line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "TEXTURE")==0){
					loadTexture(tokens[1]);
					cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "MATERIAL")==0){
					loadMaterial(tokens[1]);
					cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "OBJECT")==0){
					loadObject(tokens[1]);
					cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "CAMERA")==0){
					loadCamera();
					cout << " " << endl;
				}
			}
		}
	}
}

int scene::loadObject(string objectid){
	int id = atoi(objectid.c_str());
	if(id!=objects.size()){
		cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
		return -1;
	}else{
		cout << "Loading Object " << id << "..." << endl;
		geom newObject;
		string line;

		//load object type 
		utilityCore::safeGetline(fp_in,line);
		if (!line.empty() && fp_in.good()){
			if(strcmp(line.c_str(), "sphere")==0){
				cout << "Creating new sphere..." << endl;
				newObject.type = SPHERE;
			}else if(strcmp(line.c_str(), "cube")==0){
				cout << "Creating new cube..." << endl;
				newObject.type = CUBE;
			}else{
				//load mesh from obj file
				string objline = line;
				string name;
				string extension;
				istringstream liness(objline);
				getline(liness, name, '.');
				getline(liness, extension, '.');
				if(strcmp(extension.c_str(), "obj")==0){
					cout << "Creating new mesh..." << endl;
					cout << "Reading mesh from " << line << "... " << endl;
					obj* mesh = new obj();
					objLoader* loader = new objLoader(objline, mesh);
					delete loader;
					newObject.type = MESH;
					newObject.vertexcount = mesh->getVertexCount();
					newObject.normalcount = mesh->getNormalCount();
					newObject.facecount = mesh->getFaceCount();
					convertObj(mesh, id);
					vertexcount += mesh->getVertexCount();
					normalcount += mesh->getNormalCount();
					uvcount += mesh->getTexcoordCount();
				}else{
					cout << "ERROR: " << line << " is not a valid object type!" << endl;
					return -1;
				}
			}
		}

		//link material
		utilityCore::safeGetline(fp_in,line);
		if(!line.empty() && fp_in.good()){
			vector<string> tokens = utilityCore::tokenizeString(line);
			newObject.materialid = atoi(tokens[1].c_str());
			cout << "Connecting Object " << objectid << " to Material " << newObject.materialid << "..." << endl;
		}

		//load frames
		int frameCount = 0;
		utilityCore::safeGetline(fp_in,line);
		vector<glm::vec3> translations;
		vector<glm::vec3> scales;
		vector<glm::vec3> rotations;
		while (!line.empty() && fp_in.good()){

			//check frame number
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
				cout << "ERROR: Incorrect frame count!" << endl;
				return -1;
			}

			//load tranformations
			for(int i=0; i<3; i++){
				glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
				utilityCore::safeGetline(fp_in,line);
				tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "TRANS")==0){
					translations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
				}else if(strcmp(tokens[0].c_str(), "ROTAT")==0){
					rotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
				}else if(strcmp(tokens[0].c_str(), "SCALE")==0){
					scales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
				}
			}

			frameCount++;
			utilityCore::safeGetline(fp_in,line);
		}

		//move frames into CUDA readable arrays
		newObject.translations = new glm::vec3[frameCount];
		newObject.rotations = new glm::vec3[frameCount];
		newObject.scales = new glm::vec3[frameCount];
		newObject.transforms = new cudaMat4[frameCount];
		newObject.inverseTransforms = new cudaMat4[frameCount];
		for(int i=0; i<frameCount; i++){
			newObject.translations[i] = translations[i];
			newObject.rotations[i] = rotations[i];
			newObject.scales[i] = scales[i];
			glm::mat4 transform = utilityCore::buildTransformationMatrix(translations[i], rotations[i], scales[i]);
			newObject.transforms[i] = utilityCore::glmMat4ToCudaMat4(transform);
			newObject.inverseTransforms[i] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
		}

		objects.push_back(newObject);

		cout << "Loaded " << frameCount << " frames for Object " << objectid << "!" << endl;
		return 1;
	}
}

int scene::loadCamera(){
	cout << "Loading Camera ..." << endl;
	camera newCamera;
	float fovy;

	//load static properties
	for(int i=0; i<4; i++){
		string line;
		utilityCore::safeGetline(fp_in,line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "RES")==0){
			newCamera.resolution = glm::vec2(atoi(tokens[1].c_str()), atoi(tokens[2].c_str()));
		}else if(strcmp(tokens[0].c_str(), "FOVY")==0){
			fovy = atof(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "ITERATIONS")==0){
			newCamera.iterations = atoi(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "FILE")==0){
			newCamera.imageName = tokens[1];
		}
	}

	//load time variable properties (frames)
	int frameCount = 0;
	string line;
	utilityCore::safeGetline(fp_in,line);
	vector<glm::vec3> positions;
	vector<glm::vec3> views;
	vector<glm::vec3> ups;
	vector<float> apertures;
	vector<float> focusPlanes;
	while (!line.empty() && fp_in.good()){

		//check frame number
		vector<string> tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
			cout << "ERROR: Incorrect frame count!" << endl;
			return -1;
		}

		//load camera properties
		for(int i=0; i<5; i++){
			//glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
			utilityCore::safeGetline(fp_in,line);
			tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "EYE")==0){
				positions.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
			}else if(strcmp(tokens[0].c_str(), "VIEW")==0){
				views.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
			}else if(strcmp(tokens[0].c_str(), "UP")==0){
				ups.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
			}
			else if(strcmp(tokens[0].c_str(), "APERTURE")==0){
				apertures.push_back(atof(tokens[1].c_str()));
			}else if(strcmp(tokens[0].c_str(), "FOCUSPLANE")==0){
				focusPlanes.push_back(atof(tokens[1].c_str()));
			}

		}

		frameCount++;
		utilityCore::safeGetline(fp_in,line);
	}
	newCamera.frames = frameCount;

	//move frames into CUDA readable arrays
	newCamera.positions = new glm::vec3[frameCount];
	newCamera.views = new glm::vec3[frameCount];
	newCamera.ups = new glm::vec3[frameCount];
	newCamera.apertures = new float[frameCount];
	newCamera.focusPlanes = new float[frameCount];
	for(int i=0; i<frameCount; i++){
		newCamera.positions[i] = positions[i];
		newCamera.views[i] = views[i];
		newCamera.ups[i] = ups[i];
		newCamera.apertures[i] = apertures[i];
		newCamera.focusPlanes[i] = focusPlanes[i];
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy*(PI/180));
	float xscaled = (yscaled * newCamera.resolution.x)/newCamera.resolution.y;
	float fovx = (atan(xscaled)*180)/PI;
	newCamera.fov = glm::vec2(fovx, fovy);

	renderCam = newCamera;

	//set up render camera stuff
	renderCam.image = new glm::vec3[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	renderCam.rayList = new ray[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	for(int i=0; i<renderCam.resolution.x*renderCam.resolution.y; i++){
		renderCam.image[i] = glm::vec3(0,0,0);
	}

	cout << "Loaded " << frameCount << " frames for camera!" << endl;
	return 1;
}

int scene::loadTexture(string textureid){
	int id = atoi(textureid.c_str());
	if(id != textures.size()){
		cout << "ERROR: TEXTURE ID does not match expected number of textures" << endl;
		return -1;
	}else{
		cout << "Loading Texture " << id << "..." << endl;
		cputexture newTexture;

		// load texture from file
		string filename;
		utilityCore::safeGetline(fp_in,filename);
		
		int width, height, n;
		unsigned char* colorData = stbi_load(filename.c_str(), &width, &height, &n, 3);

		glm::vec3* colors = new glm::vec3[width * height];
		for (int y=0; y<height; ++y) {
			for (int x=0; x<width; ++x) {
				int idx = y * width + x;
				colors[idx].r = (int)colorData[idx*3]/255.0;
				colors[idx].g = (int)colorData[idx*3+1]/255.0;
				colors[idx].b = (int)colorData[idx*3+2]/255.0;
			}
		}
		
		delete [] colorData;

		newTexture.width = width;
		newTexture.height = height;
		newTexture.colors = colors;
		textures.push_back(newTexture);
		widthcount += width;
		maxheight = max(height, maxheight);
		return 1;
	}
}

int scene::loadMaterial(string materialid){
	int id = atoi(materialid.c_str());
	if(id!=materials.size()){
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}else{
		cout << "Loading Material " << id << "..." << endl;
		material newMaterial;

		//load static properties
		for(int i=0; i<11; i++){
			string line;
			utilityCore::safeGetline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "RGB")==0){
				glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.color = color;
			}else if(strcmp(tokens[0].c_str(), "SPECEX")==0){
				newMaterial.specularExponent = atof(tokens[1].c_str());				  
			}else if(strcmp(tokens[0].c_str(), "SPECRGB")==0){
				glm::vec3 specColor( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.specularColor = specColor;
			}else if(strcmp(tokens[0].c_str(), "REFL")==0){
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFR")==0){
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFRIOR")==0){
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "SCATTER")==0){
				newMaterial.hasScatter = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "ABSCOEFF")==0){
				glm::vec3 abscoeff( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.absorptionCoefficient = abscoeff;
			}else if(strcmp(tokens[0].c_str(), "RSCTCOEFF")==0){
				newMaterial.reducedScatterCoefficient = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "EMITTANCE")==0){
				newMaterial.emittance = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "TEXTURE")==0){
				newMaterial.textureid = atof(tokens[1].c_str());					  
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}

// Use the loaded mesh's information to populate the vector of faces
int scene::convertObj(obj* mesh, int geomid) {
	for (int i=0; i<mesh->getFaceCount(); ++i) {
		vector<int> facePoints = mesh->getFaces()->at(i);
		vector<int> faceNormals = mesh->getFaceNormals()->at(i);
		vector<int> faceUVs = mesh->getFaceTextures()->at(i);
		if (facePoints.size() != 3) {
			cout << "ERROR: mesh is not triangulated" << endl;
			return -1;
		}
		triangle face(geomid, facePoints[0]+vertexcount, facePoints[1]+vertexcount, facePoints[2]+vertexcount,
			faceNormals[0]+normalcount, faceNormals[1]+normalcount, faceNormals[2]+normalcount,
			faceUVs[0]+uvcount, faceUVs[1]+uvcount, faceUVs[2]+uvcount);
		faces.push_back(face);
	}

	for (int i=0; i<mesh->getVertexCount(); ++i) {
		glm::vec3 vert(mesh->getPoints()->at(i).x, mesh->getPoints()->at(i).y, mesh->getPoints()->at(i).z);
		vertices.push_back(vert);
	}

	for (int i=0; i<mesh->getNormalCount(); ++i) {
		glm::vec3 norm(mesh->getNormals()->at(i).x, mesh->getNormals()->at(i).y, mesh->getNormals()->at(i).z);
		normals.push_back(norm);
	}

	for (int i=0; i<mesh->getTexcoordCount(); ++i) {
		glm::vec2 uv(mesh->getTextureCoords()->at(i).x, mesh->getTextureCoords()->at(i).y);
		uvs.push_back(uv);
	}
	
	return 1;
}