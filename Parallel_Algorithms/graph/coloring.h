#include <cuda.h>
#include <curand_kernel.h>
#include "graph.h"
#include "/content/drive/MyDrive/graphcoloring/utils/common.h"

struct Coloring {
	bool		uncoloredNodes;
	uint		numOfColors;
	int	*		coloring;   // each element denotes a color
};


Coloring* graphColoring(GraphStruct*);
__global__ void init(uint, curandState_t*, uint*, uint);
__global__ void print_d(GraphStruct*, bool);
__global__ void colorer(Coloring*, GraphStruct *);
