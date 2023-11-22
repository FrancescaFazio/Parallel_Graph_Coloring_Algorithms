#include "graph\graph.h"
#include "graph\graph_d.h"
#include "graph\coloring.h"
#include "utils\common.h"

#define THREADxBLOCK 128

using namespace std;

__global__ void colorer(Coloring * col, GraphStruct *str){
	int n = str->nodeSize;

	for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x) {
		bool flag = true; // vera sse il nodo ha peso locale massimo

		// ignora i nodi già colorati
		if ((col->coloring[i] != -1)) continue;

		int iWeight = str->weights[i];

		// guarda i pesi del vicinato
		uint offset = str->cumDegs[i];
		uint deg = str->cumDegs[i + 1] - str->cumDegs[i];

		for (uint j = 0; j < deg; j++) {
			uint neighID = str->neighs[offset + j];
			// ignora i vicini già colorati (e te stesso)
			int jColor = col->coloring[neighID];
			if (((jColor != -1) && (jColor != col->numOfColors)) || (i == neighID)) continue;
			int jWeight = str->weights[neighID];
			if (iWeight <= jWeight) flag = false;
		}
		// colora solo se sei il nodo di peso massimo
		if (flag) col->coloring[i] = col->numOfColors;
  	}
}

void FYshuffle(int * weights, uint n){
    for(int i = 0; i < n; i++){
        int swapIdx = (rand() % (n - i)) + i;
        int tmp = weights[i];
				weights[i] = weights[swapIdx];
				weights[swapIdx] = tmp;
    }
}

Coloring* graphColoring(GraphStruct *str){
	int n = str->nodeSize;
	Coloring* col;
	CHECK(cudaMallocManaged(&col, sizeof(Coloring)));
	col->uncoloredNodes = true;

    CHECK(cudaMallocManaged(&(col->coloring), n * sizeof(int)));
	memset(col->coloring, -1 ,n * sizeof(int));
	// allocate space on the GPU for the random states

	dim3 threads ( THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

	for (int i = 0; i < n; i++){
        str->weights[i] = i;
    }
    FYshuffle(str->weights, n);

	bool flag = false;
	for(int c = 0; c < n; c++){
		col->numOfColors = c;
		colorer<<<blocks, threads>>>(col, str);
		cudaDeviceSynchronize();

		for(int i=0; i<n; i++){
				if(col->coloring[i] == -1){
						flag = true;
				}
		}

		if(!flag) break;
		else flag = false;

	}

    return col;
}
