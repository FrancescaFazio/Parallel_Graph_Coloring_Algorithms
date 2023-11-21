#include <cuda.h>
#include <iostream>
#include <curand_kernel.h>
#include "graph\graph.h"
#include "graph\graph_d.h"
#include "graph\coloring.h"
#include "utils\common.h"

#define THREADxBLOCK 128

using namespace std;

__global__ void setDegrees(GraphStruct *str, uint k, uint weight){
    uint n = str->nodeSize;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x*gridDim.x) {
        uint offset = str->cumDegs[idx];
        uint originalDeg = str->cumDegs[idx + 1] - str->cumDegs[idx];
        uint inducedDeg = 0;

        for (uint i = 0; i < originalDeg; i++){
            uint neighID = str->neighs[offset + i];
            if(str->weights[neighID] == -1) inducedDeg += 1;
        }

        // printf("Nodo %d, Grado originale : %d, Grado indotto : %d, k : %d, Peso : %d \n", idx, originalDeg, inducedDeg, k, weight);

        if (inducedDeg <= k && str->weights[idx] == -1) str->weights[idx] = weight;
    }
}

void initDegrees(GraphStruct *str){
    uint degree = 1;
    uint weigth = 1;
    uint n = str->nodeSize;

    dim3 threads ( THREADxBLOCK);
    dim3 blocks ((n + threads.x - 1) / threads.x, 1, 1 );

    bool flag = true;

    while(flag){
        for(int i = 0; i < n; i++){
            setDegrees <<< 1, 1 >>> (str, degree, weigth);
            cudaDeviceSynchronize();
            for (int i = 0; i < n; i++){
                printf("%d ", str->weights[i]);
            }
            printf("\n\n");

            flag = false;

            for(int j = 0; j < n; j++){
                if(str->weights[j] == -1) flag = true;
		    }
            if(!flag) break;
            weigth += 1;
        }
        degree += 1; 
    }
}

__global__ void findCandidates (Coloring* col, GraphStruct *str, uint* degrees, uint* weigths, bool* candidateNodes) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= str->nodeSize)
		return;

	if (col->coloring[idx])
		return;

	uint offset = str->cumDegs[idx];
	uint deg = str->cumDegs[idx + 1] - str->cumDegs[idx];

	bool candidate = true;
    for (uint j = 0; j < deg; j++) {
	    uint neighID = str->neighs[offset + j];
		if (col->coloring[neighID] == 0 &&
				((degrees[idx] < degrees[neighID]) ||
				((degrees[idx] = degrees[neighID]) && (weigths[idx] < weigths[neighID])))) {
			candidate = false;
		}
	}

    if(candidate){
        candidateNodes[idx] = true;
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
        int minColor = n; //colore minimo disponibile tra i vicini

		for (uint j = 0; j < deg; j++) {
			uint neighID = str->neighs[offset + j];
			// ignora i vicini già colorati (e te stesso)
			int jColor = col->coloring[neighID];
            if (jColor < minColor) minColor = jColor;
            if (jColor != -1 || i == neighID) continue;
			int jWeight = str->weights[neighID];
			if (iWeight <= jWeight) flag = false;
		}

        printf("Min color: %d\n", minColor);
		// colora solo se sei il nodo di peso massimo
		if (flag){
            if(minColor == -1 || minColor > 0) col->coloring[i] = 0;
        }
	}
}

void h_swap(int* array, int idx_a, int idx_b){
    int tmp = array[idx_a];
    array[idx_a] = array[idx_b];
    array[idx_b] = tmp;

    return;
}

void FYshuffle(int * weights, uint n){
    for(int i = 0; i < n; i++){
        int swapIdx = (rand() % (n - i)) + i;
        h_swap(weights, i, swapIdx);
    }
}

Coloring* graphColoring(GraphStruct *str) {
	int n = str->nodeSize;
	Coloring* col;
	CHECK(cudaMallocManaged(&col, sizeof(Coloring)));
	col->uncoloredNodes = true;

    CHECK(cudaMallocManaged(&(col->coloring), n * sizeof(int)));
	memset(col->coloring, -1 ,n * sizeof(int));
	// allocate space on the GPU for the random states

	dim3 threads ( THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

    initDegrees(str);
	for (int i = 0; i < n; i++){
        printf("%d ", str->weights[i]);
    }
    printf("\n");   
    
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