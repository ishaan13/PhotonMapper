NVCC     = /usr/local/cuda/bin/nvcc -m64
CC       = /usr/bin/gcc -m64

GLFW_INCLUDE_PATH = -Iglfw/include/ 
GLFW_LIB_PATH = -Lglfw/lib/
GLFW = $(GLFW_INCLUDE_PATH) $(GLFW_LIB_PATH)

CUDA_INCLUDE = -I/usr/local/cuda/include  
CUDASDK_C_LIB_PATH = -L/Developer/GPU\ Computing/C/lib
CUDASDK_C_INCLUDE_PATH = -I/Developer/GPU\ Computing/C/common/inc
CUDA = $(CUDA_INCLUDE) $(CUDASDK_C_LIB_PATH) $(CUDASDK_C_INCLUDE_PATH) 

XLINKER = -Xlinker -framework,OpenGL,-framework,GLUT

LFLAGS = $(GLFW) $(CUDA) $(XLINKER) -lglfw 

all: 565raytracer

565raytracer: ../src/main.cpp
	$(NVCC) $(GLEW_PATH) $(LFLAGS) ../src/main.cpp ../src/raytraceKernel.cu ../src/glslUtility.cpp ../src/utilities.cpp ../src/image.cpp ../src/scene.cpp ../src/stb_image/stb_image_write.c ../src/stb_image/stb_image.c -o bin/565raytracer

clean:
	rm bin/565raytracer
	rm *.o

.PHONY : bin/565raytracer