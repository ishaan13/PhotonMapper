// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  liveCamera.aperture = 0;//1.75;
  liveCamera.focusPlane = 0;//12.5;

  #ifdef __APPLE__
	  // Needed in OSX to force use of OpenGL3.2 
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
	  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #endif

  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = true;

  // Load scene file
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "scene")==0){
      renderScene = new scene(data);
      loadedScene = true;
    }else if(strcmp(header.c_str(), "frame")==0){
      targetFrame = atoi(data.c_str());
      singleFrameMode = true;
    }
  }

  if(!loadedScene){
    cout << "Error: scene file needed!" << endl;
    return 0;
  }

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];

  if(targetFrame>=renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

  // Launch CUDA/GL

  #ifdef __APPLE__
	init();
  #else
	init(argc, argv);
  #endif

  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
	  // send into GLFW main loop
	  while(1){
		display();
		if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
				exit(0);
		}
	  }

	  glfwTerminate();
  #else
	  glutDisplayFunc(display);
	  glutKeyboardFunc(keyboard);
	  glutSpecialFunc(specialKeyboard);

	  glutMainLoop();
  #endif

  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){

	
	// Performance Analysis End
	cudaEvent_t start,stop;
	// Generate events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	// Trigger event 'start'
	cudaEventRecord(start, 0);

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

  if(iterations<renderCam->iterations){
    uchar4 *dptr=NULL;
    iterations++;
    cudaGLMapBufferObject((void**)&dptr, pbo);
  
    // execute the kernel
   //cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size(), liveCamera);
   cudaPhotonMapCore(renderCam, targetFrame, iterations, dptr, liveCamera);

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  }else{

    if(!finishedRender){
      //output image file
      image outputImage(renderCam->resolution.x, renderCam->resolution.y);

      for(int x=0; x<renderCam->resolution.x; x++){
        for(int y=0; y<renderCam->resolution.y; y++){
          int index = x + (y * renderCam->resolution.x);
          outputImage.writePixelRGB(renderCam->resolution.x-1-x,y,renderCam->image[index]);
        }
      }
      
      gammaSettings gamma;
      gamma.applyGamma = true;
      gamma.gamma = 1.0;//2.2;
      gamma.divisor = renderCam->iterations;
      outputImage.setGammaSettings(gamma);
      string filename = renderCam->imageName;
      string s;
      stringstream out;
      out << targetFrame;
      s = out.str();
      utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
      utilityCore::replaceString(filename, ".png", "."+s+".png");
      outputImage.saveImageRGB(filename);
      cout << "Saved frame " << s << " to " << filename << endl;
      finishedRender = true;
      if(singleFrameMode==true){
        cudaDeviceReset();
        exit(0);
      }
    }
    if(targetFrame<renderCam->frames-1){

      //clear image buffer and move onto next frame
      targetFrame++;
      iterations = 0;
      for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
        renderCam->image[i] = glm::vec3(0,0,0);
      }
      cudaDeviceReset(); 
      finishedRender = false;
    }
  }
	// Performance Analysis End
	cudaEventRecord(stop, 0); // Trigger Stop event
	cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)
	float elapsedTime; // Initialize elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
 
	printf("Execution Time: %fms for iteration: %d\n", elapsedTime, iterations); // Print Elapsed time
}

#ifdef __APPLE__

	void display(){
		runCuda();

		string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glfwSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glfwSwapBuffers();
	}

#else

	void display(){

		runCuda();

		string title = "MMFAPhoMap | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glutSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glutPostRedisplay();
		glutSwapBuffers();
	}

	void resetAccumulator()
	{
		iterations = 1;
	}

	void keyboard(unsigned char key, int x, int y)
	{
//		std::cout << key << std::endl;
		switch (key) 
		{
		   case(27):
			   exit(1);
			   break;
		   case('w'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.position.y += STRAFE_AMOUNT;
			   break;
		   case('s'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.position.y -= STRAFE_AMOUNT;
			   break;
		   case('a'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.position.x += STRAFE_AMOUNT;
			   break;
		   case('d'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.position.x -= STRAFE_AMOUNT;
			   break;
		   case('q'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.position.z += STRAFE_AMOUNT;
			   break;
		   case('z'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.position.z -= STRAFE_AMOUNT;
			   break;
		   case('['):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.view.x += STRAFE_AMOUNT;
			   break;
		   case(']'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.view.x -= STRAFE_AMOUNT;
			   break;
		   case('o'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.view.y += STRAFE_AMOUNT;
			   break;
		   case('p'):
			   iterations = 1;
			   cudaClearAccumulatorImage(renderCam);
			   liveCamera.view.y -= STRAFE_AMOUNT;
			   break;
		}
	}

	void specialKeyboard(int key, int x, int y)
	{
			switch (key) 
			{
				case GLUT_KEY_UP:
					iterations = 1;
					cudaClearAccumulatorImage(renderCam);
					liveCamera.focusPlane += STRAFE_AMOUNT;
					break;
				case GLUT_KEY_DOWN:
					iterations = 1;
					cudaClearAccumulatorImage(renderCam);
					liveCamera.focusPlane -= STRAFE_AMOUNT;
					break;
				case GLUT_KEY_LEFT:
					iterations = 1;
					cudaClearAccumulatorImage(renderCam);
					liveCamera.aperture += 0.5*STRAFE_AMOUNT;
					break;
				case GLUT_KEY_RIGHT:
					iterations = 1;
					cudaClearAccumulatorImage(renderCam);
					liveCamera.aperture -= 0.5*STRAFE_AMOUNT;
					if(liveCamera.aperture < NUDGE)
						liveCamera.aperture = NUDGE;
					break;
			}
	}

#endif




//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init(){

		if (glfwInit() != GL_TRUE){
			shut_down(1);      
		}

		// 16 bit color, no depth, alpha or stencil buffers, windowed
		if (glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW) != GL_TRUE){
			shut_down(1);
		}

		// Set up vertex array object, texture stuff
		initVAO();
		initTextures();
	}
#else
	void init(int argc, char* argv[]){
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow("565Raytracer");

		// Init GLEW
		glewInit();
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			/* Problem: glewInit failed, something is seriously wrong. */
			std::cout << "glewInit failed, aborting." << std::endl;
			exit (1);
		}

		initVAO();
		initTextures();
	}
#endif

void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);
  atexit(cudaFreeMemory);

      //pack geom and material arrays
    geoms = new geom[renderScene->objects.size()];
    materials = new material[renderScene->materials.size()];
    
    for(int i=0; i<renderScene->objects.size(); i++){
      geoms[i] = renderScene->objects[i];
    }
    for(int i=0; i<renderScene->materials.size(); i++){
      materials[i] = renderScene->materials[i];
    }

	 //copy stuff to the gpu
	cudaAllocateMemory(renderCam, materials, renderScene->materials.size(), geoms, renderScene->objects.size());

	delete[] geoms;
	delete[] materials;

  runCuda();
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  #ifdef __APPLE__
	glfwTerminate();
  #endif
  exit(return_code);
}
