#include <cuda.h>
#include <curand_kernel.h>
#include "graph.h"
#include "..\utils\common.h"

struct Coloring {
	// bool *		usedColors;		// maschera booleana dei colori usati dai vicini (per LDF e SDL)
	uint		numOfColors;
	int	*		coloring;   	
	bool 		uncoloredNodes;
};


Coloring* graphColoring(GraphStruct*);
//void CPUcolorer(Coloring*, GraphStruct *);
//__global__ void init(uint, curandState_t*, uint*, uint);
__global__ void print_d(GraphStruct*, bool);
//__global__ void colorer(Coloring*, GraphStruct *);
