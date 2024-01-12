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

__global__ void colorer(Coloring * col, GraphStruct * str){
    uint n = str->nodeSize;

    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x) {
        bool flag = true; // vera sse il nodo ha peso locale massimo

        // ignora i nodi già colorati
        if ((col->coloring[i] != -1)) return;

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

	// Creazione coloratura CPU e GPU
	Coloring * col_h;
	Coloring * col_d;
	int * coloring_d;

    // Generazione pesi
    thrust::sequence(str->weights, str->weights + n);
    thrust::default_random_engine g;
    thrust::shuffle(str->weights, str->weights + n, g);

	// CPU
	col_h = (Coloring *) malloc(sizeof(Coloring));

	col_h->coloring = (int *) malloc(n * sizeof(int));
	thrust::fill(col_h->coloring, col_h->coloring + n, -1);

	col_h->numOfColors = 0;
	col_h->uncoloredNodes = true;

	// GPU
	gpuErrchk(cudaMalloc((void **) &col_d, sizeof(Coloring)));
	gpuErrchk(cudaMalloc((void **) &coloring_d, n * sizeof(int)));

	gpuErrchk(cudaMemcpy(coloring_d, col_h->coloring, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&(col_d->coloring), &coloring_d, sizeof(col_d->coloring), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(&(col_d->numOfColors), 0, sizeof(uint)));
	gpuErrchk(cudaMemset(&(col_d->uncoloredNodes), false, sizeof(bool)));

	dim3 threads (THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start);

	while(col_h->uncoloredNodes){
		col_h->uncoloredNodes = false;
		col_h->numOfColors++;

		// Aggiorno coloring GPU
		gpuErrchk(cudaMemcpy(&(col_d->uncoloredNodes), &(col_h->uncoloredNodes), sizeof(bool), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(&(col_d->numOfColors), &(col_h->numOfColors), sizeof(uint), cudaMemcpyHostToDevice));

		colorer<<<blocks, threads>>>(col_d, str);

        //gpuErrchk(cudaPeekAtLastError());
        //gpuErrchk(cudaDeviceSynchronize());

		// Aggiorno uncoloredNodes lato CPU
		gpuErrchk(cudaMemcpy(&(col_h->uncoloredNodes), &(col_d->uncoloredNodes), sizeof(bool), cudaMemcpyDeviceToHost));
	}

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo solo kernel %f ms\n", milliseconds);

	// Copio il risultato da CPU a GPU
	gpuErrchk(cudaMemcpy(col_h->coloring, coloring_d, n * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&(col_h->numOfColors), &(col_d->numOfColors), sizeof(uint), cudaMemcpyDeviceToHost));

    return col_h;
}