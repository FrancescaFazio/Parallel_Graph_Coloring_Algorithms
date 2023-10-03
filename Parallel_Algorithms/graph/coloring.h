#include "curand_kernel.h"
#include "/content/drive/MyDrive/graph/common.h"
#include "/content/drive/MyDrive/graph/graph.h"


struct Coloring {
	bool		uncoloredNodes;
	uint		numOfColors;
	uint	*	coloring;   // each element denotes a color
};


Coloring* graphColoring(GraphStruct*);
__global__ void init(uint, curandState_t*, uint*, uint);
__global__ void print_d(GraphStruct*, bool);
__global__ void colorer(Coloring*, GraphStruct *, bool *);