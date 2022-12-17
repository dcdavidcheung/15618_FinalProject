CXX=g++ -m64
CXXFLAGS=-I../common -Iobjs/ -O3 -Wall
ISPC=ispc
# note: change target to sse4 for SSE capable machines
ISPCFLAGS=-O2 --target=avx1-i32x8 --arch=x86-64


scene_ispc.h: scene.ispc
	$(ISPC) $(ISPCFLAGS) $< -o ./scene_ispc.o -h ./scene_ispc.h
