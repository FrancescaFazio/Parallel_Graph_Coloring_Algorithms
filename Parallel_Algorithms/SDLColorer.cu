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

#define THREADxBLOCK 128
#define GRAPH_DIM 1000

using namespace std;

__device__ int random;

__global__ void setDegrees(GraphStruct *str, uint k, uint weight){
    uint n = str->nodeSize;

    for (int idx = threadIdx.x+blockIdx.x*blockDim.x; idx < n; idx += blockDim.x*gridDim.x){
        uint offset = str->cumDegs[idx];
        uint originalDeg = str->cumDegs[idx + 1] - str->cumDegs[idx];
        uint inducedDeg = 0;

        for (uint i = 0; i < originalDeg; i++){
            uint neighID = str->neighs[offset + i];
            if(str->weights[neighID] == -1) inducedDeg += 1;
        }

        if (inducedDeg <= k && str->weights[idx] == -1) str->weights[idx] = weight;
    }
}

void initDegrees(GraphStruct *str){
    uint degree = 1;
    uint weigth = 1;
    uint n = str->nodeSize;

    dim3 threads ( THREADxBLOCK);
    dim3 blocks ((n + threads.x - 1) / threads.x, 1, 1 );

    int nodesLeft = n;

    while(nodesLeft > 0){
        while(true){
            setDegrees <<< blocks, threads >>> (str, degree, weigth);
            gpuErrchk(cudaPeekAtLastError())
		    gpuErrchk(cudaDeviceSynchronize());

            int visitedNodes = (int)thrust::count(str->weights, str->weights+n, weigth);

            if(visitedNodes == 0) break;
            weigth += 1;
            nodesLeft -= visitedNodes;
        }
        degree += 1; 
    }
}

__global__ void colorer (Coloring* col, GraphStruct *str) {
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

            if(jColor != -1){
                col->usedColors[n * i + jColor] = true;
                continue;
            }
            if(i == neighID) continue;
			int jWeight = str->weights[neighID];
            
			if (iWeight < jWeight) flag = false;
            else if (iWeight == jWeight){
                int iRandom = (i + random) % n;
                int jRandom = (neighID + random) % n;
                if(iRandom < jRandom) flag = false;
            }
		}
           
		// colora solo se sei il nodo di peso massimo
		if (flag){
            // Cerca il primo colore libero per questo nodo
            int color = 0;
            while (col->usedColors[n * i + color]) color++;
            
            // Assegna il primo colore libero al nodo corrente
            col->coloring[i] = color;
        }
    }
}


Coloring* graphColoring(GraphStruct *str) {
	int n = str->nodeSize;
    int r = rand();
    
    cudaMemcpyToSymbol(random, &r, sizeof(int));

	Coloring* col;
	gpuErrchk(cudaMallocManaged(&col, sizeof(Coloring)));

    gpuErrchk(cudaMallocManaged(&(col->coloring), n * sizeof(int)));
	thrust::fill(col->coloring, col->coloring + n, -1);
	// allocate space on the GPU for the random states

    gpuErrchk(cudaMallocManaged(&(col->usedColors), n * n * sizeof(bool)));
    thrust::fill(col->usedColors, col->usedColors + (n * n), false);

	dim3 threads ( THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

    initDegrees(str);
    gpuErrchk(cudaPeekAtLastError())
    gpuErrchk(cudaDeviceSynchronize());

	for(int c = 0; c < n; c++){
		colorer<<<blocks, threads>>>(col, str);
        gpuErrchk(cudaPeekAtLastError())
		gpuErrchk(cudaDeviceSynchronize());
        
        int left = (int)thrust::count(col->coloring, col->coloring + n ,-1);
        if (left == 0) break;
	}

    return col;
}