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

__global__ void warm_up_gpu(){
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid; 
}

__global__ void findCandidates(Coloring * col, GraphStruct * str, bool * currentIS, bool * usedColors){
    uint n = str->nodeSize;
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int jColor, jWeight, neighID;

    if (i > n) return; 

    bool flag = true; 

    // ignora i nodi già colorati
    if ((col->coloring[i] != -1)) return;

    int iWeight = str->weights[i];

    // guarda i pesi del vicinato
    uint offset = str->cumDegs[i];
    uint deg = str->cumDegs[i + 1] - str->cumDegs[i];
    
    for (uint j = 0; j < deg; j++) {
        neighID = str->neighs[offset + j];
        jColor = col->coloring[neighID];

        jWeight = str->weights[neighID];
        if (!((jColor != -1)  || (i == neighID)) && iWeight <= jWeight) flag = false;
    }

    // colora solo se sei il nodo di peso massimo
    if (flag) currentIS[i] = true;
    
}

__global__ void colorer(Coloring * col, GraphStruct *str, bool * currentIS, bool * usedColors){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    uint n = str->nodeSize;

    if (i >= n) return;

    if (currentIS[i] == 1 && col->coloring[i] == -1){
        uint offset = str->cumDegs[i];
		int deg = str->cumDegs[i + 1] - str->cumDegs[i];
        uint neighID; int jColor; bool flag = true; int c = 0;

        while(flag){
            flag = false;
            for (uint j = 0; j < deg; j++) {
                neighID = str->neighs[offset + j];
                jColor = col->coloring[neighID];

                if (jColor == c){
                    flag = true;
                    c++;
                    break;
                }
            }
        }

        col->coloring[i] = c;
    }else if(currentIS[i] == 0 && col->coloring[i] == -1){
        col->uncoloredNodes = true;
    }
}

Coloring* graphColoring(GraphStruct *str) {
	int n = str->nodeSize;
    bool * currentIS;
	printf("%d %d\n",n, str->edgeSize);

	// Creazione coloratura CPU e GPU
	Coloring * col_h;
	Coloring * col_d;
	int * coloring_d;
    bool * usedColors_d;

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
    gpuErrchk(cudaMalloc((void **) &currentIS, n * sizeof(bool)));
	gpuErrchk(cudaMalloc((void **) &coloring_d, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &usedColors_d, str->edgeSize * sizeof(bool)));

	gpuErrchk(cudaMemcpy(coloring_d, col_h->coloring, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&(col_d->coloring), &coloring_d, sizeof(col_d->coloring), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(&(col_d->numOfColors), 0, sizeof(uint)));
	gpuErrchk(cudaMemset(&(col_d->uncoloredNodes), false, sizeof(bool)));
    gpuErrchk(cudaMemset(currentIS, false, n * sizeof(bool)));
    gpuErrchk(cudaMemset(usedColors_d, false, str->edgeSize * sizeof(bool)));

	dim3 threads (THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

    warm_up_gpu<<<blocks, threads>>>();

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

		findCandidates<<<blocks, threads>>>(col_d, str, currentIS, usedColors_d);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        colorer<<<blocks, threads>>>(col_d, str, currentIS, usedColors_d);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
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