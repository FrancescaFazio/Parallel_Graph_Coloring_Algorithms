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
#define NSTREAM 2

using namespace std;

__global__ void colorer(Coloring * col, uint iElem, int ioffset, uint * cumDegs, uint * neighs, int * weights){
	int i = threadIdx.x+blockIdx.x*blockDim.x;

	if(i >= iElem) return;
	
	bool flag = true; // vera sse il nodo ha peso locale massimo

	// ignora i nodi già colorati
	if ((col->coloring[ioffset + i] != -1)) return;

	int iWeight = weights[ioffset + i];

	// guarda i pesi del vicinato
	uint offset = cumDegs[i] - cumDegs[0];
	uint deg = cumDegs[i + 1] - cumDegs[i];
	
	for (uint j = 0; j < deg; j++) {
		// printf("%d\n", offset + j);
		uint neighID = neighs[offset + j];
		// ignora i vicini già colorati (e te stesso)
		int jColor = col->coloring[neighID];
		if (((jColor != -1) && (jColor != col->numOfColors)) || ((ioffset + i) == neighID)) continue;
		//printf("Nodo %d di peso %d e grado %d. Vicino: %d di peso %d\n", ioffset + i, iWeight, deg, neighID, weights[neighID]);
		int jWeight = weights[neighID];
		// printf("%d < %d \n", iWeight, jWeight);
		if (iWeight <= jWeight) flag = false;
	}

	//printf("flag: %d", flag);
	// colora solo se sei il nodo di peso massimo
	if (flag) col->coloring[ioffset + i] = col->numOfColors;
	else col->uncoloredNodes = true;
}

Coloring* graphColoring(GraphStruct *str) {
	int n = str->nodeSize;

	// Parametri stream
	int iElem = n / NSTREAM;
	
	cudaStream_t stream[NSTREAM];

	for (int i = 0; i < NSTREAM; ++i)
		gpuErrchk(cudaStreamCreate(&stream[i]));

	// Creazione coloratura CPU e GPU
	Coloring * col_h;
	Coloring * col_d;
	int * coloring_d;
	uint * cumDegs_d;
	uint * neights_d;
	int * weights_d;

    // Generazione pesi
    thrust::sequence(str->weights, str->weights + n);
    thrust::default_random_engine g;
    thrust::shuffle(str->weights, str->weights + n, g);

	// CPU
	col_h = (Coloring *) malloc(sizeof(Coloring));

	col_h->coloring = (int *) malloc(n * sizeof(int));
	thrust::fill(col_h->coloring, col_h->coloring + n, -1);

    col_h->usedColors = (bool *) malloc(n * sizeof(bool));
    thrust::fill(col_h->usedColors, col_h->usedColors + n, false);

	col_h->numOfColors = 0;
	col_h->uncoloredNodes = true;

	// GPU
	gpuErrchk(cudaMalloc((void **) &col_d, sizeof(Coloring)));
	gpuErrchk(cudaMalloc((void **) &coloring_d, n * sizeof(int)));
	gpuErrchk(cudaMalloc((void **) &cumDegs_d, (n+1) * sizeof(uint)));
	gpuErrchk(cudaMalloc((void **) &weights_d, n * sizeof(int)));
	gpuErrchk(cudaMalloc((void **) &neights_d, str->edgeSize * sizeof(uint)));

	gpuErrchk(cudaMemcpy(coloring_d, col_h->coloring, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(neights_d, str->neighs, str->edgeSize * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(weights_d, str->weights, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cumDegs_d, str->cumDegs, (n+1) * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&(col_d->coloring), &coloring_d, sizeof(col_d->coloring), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(&(col_d->numOfColors), 0, sizeof(uint)));
	gpuErrchk(cudaMemset(&(col_d->uncoloredNodes), false, sizeof(bool)));

	dim3 threads (THREADxBLOCK);
	dim3 blocks ((iElem + threads.x - 1) / threads.x, 1, 1 );

	int neighsOffset[NSTREAM];
	neighsOffset[0] = 0;
	for(int j = 1; j < NSTREAM; j++){
		neighsOffset[j] = neighsOffset[j-1] + (str->cumDegs[j * iElem] - str->cumDegs[(j-1) * iElem]);
	}

	while(col_h->uncoloredNodes){
		//printf("ciao");
		col_h->uncoloredNodes = false;
		col_h->numOfColors++;

		// Aggiorno coloring GPU
		gpuErrchk(cudaMemcpy(&(col_d->uncoloredNodes), &(col_h->uncoloredNodes), sizeof(bool), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(&(col_d->numOfColors), &(col_h->numOfColors), sizeof(uint), cudaMemcpyHostToDevice));

		for (int i = 0; i < NSTREAM; ++i) {
			int ioffset = i * iElem;

			colorer<<<blocks, threads, 0, stream[i]>>>(col_d, iElem, ioffset, &cumDegs_d[ioffset], &neights_d[neighsOffset[i]], weights_d);
  		}

		for (int i = 0; i < NSTREAM; ++i) {
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaStreamSynchronize(stream[i]));
  		}

		// Aggiorno uncoloredNodes lato CPU
		gpuErrchk(cudaMemcpy(&(col_h->uncoloredNodes), &(col_d->uncoloredNodes), sizeof(bool), cudaMemcpyDeviceToHost));
	}

	// Copio il risultato da CPU a GPU
	gpuErrchk(cudaMemcpy(col_h->coloring, coloring_d, n * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&(col_h->numOfColors), &(col_d->numOfColors), sizeof(uint), cudaMemcpyDeviceToHost));

    return col_h;
}