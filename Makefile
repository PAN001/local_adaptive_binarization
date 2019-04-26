EXECUTABLE := binarizewolfjolion_cuda
CU_FILES   := binarizewolfjolion.cu
CU_DEPS    :=

CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61
OBJS=binarizewolfjolion.o

all:
	g++ -I/afs/cs.cmu.edu/academic/class/15418-s19/public/projects/park-chen/local/include/opencv binarizewolfjolion.cpp timing.cpp -o binarizewolfjolion `pkg-config opencv --libs` -lstdc++

clean:
	rm -f binarizewolfjolion

cuda: $(EXECUTABLE)


test:
	./binarizewolfjolion -k 0.6 sample.jpg _result.jpg


package:	clean
	rm -f x.jpg
	tar cvfz binarizewolfjolionopencv.tgz *

$(EXECUTABLE):
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) `pkg-config opencv --libs` -lstdc++


g++ -m64 -O3 -Wall -o binarizewolfjolion.o -L/usr/local/depot/cuda-8.0/lib64/ -lcudart `pkg-config opencv --libs` -lstdc++
