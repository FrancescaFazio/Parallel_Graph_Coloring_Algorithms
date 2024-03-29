#include <cuda.h>
#include <iostream>
#include <curand_kernel.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdlib.h>
#include <curand.h>
#include "graph\graph.h"
#include "graph\graph_d.h"
#include "graph\coloring.h"
#include "utils\common.h"
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/count.h>

#define THREADxBLOCK 128
#define GRAPH_DIM 1000

using namespace std;

__global__ void warm_up_gpu(){
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid; 
}

__global__ void setDegrees(GraphStruct *str, uint * k, uint * weight, bool * flag, uint * visitedNodes){
    uint n = str->nodeSize;

    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx < n; idx += blockDim.x*gridDim.x){
        uint offset = str->cumDegs[idx];
        uint originalDeg = str->cumDegs[idx + 1] - str->cumDegs[idx];
        uint inducedDeg = 0;

        for (uint i = 0; i < originalDeg; i++){
            uint neighID = str->neighs[offset + i];
            if(str->weights[neighID] == -1) inducedDeg += 1;
        }

        if (inducedDeg <= * k && str->weights[idx] == -1){
            str->weights[idx] = * weight;
            * flag = true;
            * visitedNodes += 1;
        } 
    }
}

void initDegrees(GraphStruct *str){
    // CPU
    uint n = str->nodeSize;

    dim3 threads ( THREADxBLOCK);
    dim3 blocks ((n + threads.x - 1) / threads.x, 1, 1 );

    int nodesLeft = n;
    bool nodeSeen_h;
    uint visitedNodes_h;
    uint degree, weight;
    degree = 1; weight = 1;

    nodeSeen_h = true;
    visitedNodes_h = 0;

    // GPU
    bool * nodeSeen_d;
    uint * degree_d;
    uint * weigth_d;
    uint * visitedNodes_d;

    gpuErrchk(cudaMalloc((void **) &nodeSeen_d, sizeof(bool)));
    gpuErrchk(cudaMalloc((void **) &degree_d, sizeof(uint)));
    gpuErrchk(cudaMalloc((void **) &weigth_d, sizeof(uint)));
    gpuErrchk(cudaMalloc((void **) &visitedNodes_d, sizeof(uint)));

    gpuErrchk(cudaMemset(nodeSeen_d, true, sizeof(bool)));
    gpuErrchk(cudaMemset(degree_d, 1, sizeof(uint)));
    gpuErrchk(cudaMemset(weigth_d, 1, sizeof(uint)));
    gpuErrchk(cudaMemset(visitedNodes_d, 0, sizeof(uint)));

    while(nodesLeft > 0){
        while(nodeSeen_h){
            // Aggiorno flag
            gpuErrchk(cudaMemset(nodeSeen_d, false, sizeof(bool)));

            setDegrees <<< blocks, threads >>> (str, degree_d, weigth_d, nodeSeen_d, visitedNodes_d);
            gpuErrchk(cudaPeekAtLastError())
		    gpuErrchk(cudaDeviceSynchronize());

            weight += 1;

            gpuErrchk(cudaMemcpy(&nodeSeen_h, nodeSeen_d, sizeof(bool), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(&visitedNodes_h, visitedNodes_d, sizeof(uint), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(weigth_d, &weight, sizeof(uint), cudaMemcpyHostToDevice));

            nodesLeft -= visitedNodes_h;
        }
        nodeSeen_h = true;
        degree += 1; 
        gpuErrchk(cudaMemcpy(degree_d, &degree, sizeof(uint), cudaMemcpyHostToDevice));
    }
}

__global__ void findCandidates (Coloring* col, GraphStruct *str, bool * currentIS, int * weights) {
    uint n = str->nodeSize;
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int jColor, jWeight, neighID, neighDeg;

    if (i > n) return; 

    bool flag = true; 

    // ignora i nodi già colorati
    if ((col->coloring[i] != -1)) return;

    int iWeight = weights[i];

    // guarda i pesi del vicinato
    uint offset = str->cumDegs[i];
    uint deg = str->cumDegs[i + 1] - str->cumDegs[i];
    
    for (uint j = 0; j < deg; j++) {
        neighID = str->neighs[offset + j];
		neighDeg = str->cumDegs[neighID + 1] - str->cumDegs[neighID];
        jColor = col->coloring[neighID];

        jWeight = weights[neighID];
        if (!((jColor != -1)  || (i == neighID)) && (deg < neighDeg || (deg == neighDeg && iWeight < jWeight))) flag = false;
    }

    if (flag) currentIS[i] = true;
}

__global__ void colorer(Coloring * col, GraphStruct *str, bool * currentIS){
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
    int r = rand();
    bool * currentIS;
    int * weights = (int *) malloc(n * sizeof(int));
	printf("%d\n",n);

	// Creazione coloratura CPU e GPU
	Coloring * col_h;
	Coloring * col_d;
	int * coloring_d;
    int * weights_d;

    // Generazione pesi
    thrust::sequence(weights, weights + n);
    thrust::default_random_engine g;
    thrust::shuffle(weights, weights + n, g);
    initDegrees(str);

	// CPU
	col_h = (Coloring *) malloc(sizeof(Coloring));

	col_h->coloring = (int *) malloc(n * sizeof(int));
	thrust::fill(col_h->coloring, col_h->coloring + n, -1);


	col_h->numOfColors = 0;
	col_h->uncoloredNodes = true;

	// GPU
	gpuErrchk(cudaMalloc((void **) &col_d, sizeof(Coloring)));
	gpuErrchk(cudaMalloc((void **) &coloring_d, n * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &currentIS, n * sizeof(bool)));
    gpuErrchk(cudaMalloc((void **) &weights_d, n * sizeof(int)));

	gpuErrchk(cudaMemcpy(coloring_d, col_h->coloring, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&(col_d->coloring), &coloring_d, sizeof(col_d->coloring), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(weights_d, weights, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(&(col_d->numOfColors), 0, sizeof(uint)));
	gpuErrchk(cudaMemset(&(col_d->uncoloredNodes), false, sizeof(bool)));
    gpuErrchk(cudaMemset(currentIS, false, n * sizeof(bool)));
	

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

		findCandidates<<<blocks, threads>>>(col_d, str, currentIS, weights_d);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        colorer<<<blocks, threads>>>(col_d, str, currentIS);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

		// Aggiorno uncoloredNodes lato CPU e resetto currentIS
		gpuErrchk(cudaMemcpy(&(col_h->uncoloredNodes), &(col_d->uncoloredNodes), sizeof(bool), cudaMemcpyDeviceToHost));
        //gpuErrchk(cudaMemset(currentIS, false, n * sizeof(bool)));
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


			