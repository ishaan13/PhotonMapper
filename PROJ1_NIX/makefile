#All cuda paths are for v5.0+

NVCC     = nvcc -m64
CC       = gcc -m64
CPPC     = g++ -m64

CUDA_FLAGS = -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include

LFLAGS = -lglut -lGL -lGLEW 

all: 565Pathtracer

raytraceKernel.o: ../src/raytraceKernel.cu
	$(NVCC) $(CUDA_FLAGS) -c ../src/raytraceKernel.cu

glslUtility.o: ../src/glslUtility.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/glslUtility.cpp -c

utilities.o: ../src/utilities.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/utilities.cpp -c

image.o: ../src/image.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/image.cpp -c

scene.o: ../src/scene.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/scene.cpp -c

stb_image.o: ../src/stb_image/stb_image.c
	$(CC) $(CUDA_FLAGS) ../src/stb_image/stb_image.c -c

stb_image_write.o: ../src/stb_image/stb_image_write.c
	$(CC) $(CUDA_FLAGS) ../src/stb_image/stb_image_write.c -c

main.o: ../src/main.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/main.cpp -c

565Pathtracer: main.o raytraceKernel.o glslUtility.o utilities.o image.o scene.o stb_image_write.o stb_image.o
	$(NVCC) $(LFLAGS) main.o raytraceKernel.o glslUtility.o utilities.o image.o scene.o stb_image_write.o stb_image.o -o 565Pathtracer

clean:
	rm *.o
	rm 565Pathtracer

test: 565Pathtracer
	./565Pathtracer scene=../scenes/sampleScene.txt
