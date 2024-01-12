#include <cuda.h>
#include <iostream>
#include <curand_kernel.h>
#include "graph\graph.h"
#include "graph\graph_d.h"
#include "graph\coloring.h"
#include "utils\common.h"
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/count.h>

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
        else col->uncoloredNodes = true;
  	}
}

Coloring* graphColoring(GraphStruct *str) {
	int n = str->nodeSize;
    printf("%d ",n);

	Coloring* col;
	gpuErrchk(cudaMallocManaged(&col, sizeof(Coloring)));

    gpuErrchk(cudaMallocManaged(&(col->coloring), n * sizeof(int)));
	thrust::fill(col->coloring, col->coloring + n, -1);

    // Generazione pesi
    thrust::sequence(str->weights, str->weights + n);
    thrust::default_random_engine g;
    thrust::shuffle(str->weights, str->weights + n, g);

	dim3 threads ( THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

    col->uncoloredNodes = true;
    col->numOfColors = 0;
	while(col->uncoloredNodes){
        col->uncoloredNodes = false;
        colorer<<<blocks, threads>>>(col, str);
        gpuErrchk(cudaPeekAtLastError())
        gpuErrchk(cudaDeviceSynchronize());
        col->numOfColors++;

	}

    return col;
}