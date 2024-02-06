#include <iostream>
#include <cstdio>
#include "graph\graph.h"
#include "graph\graph_d.h"
#include "graph\coloring.h"
#include "utils\common.h"
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/count.h>

using namespace std;

void CPUcolorer(Coloring * col, GraphStruct *str, bool* usedColors){
	int n = str->nodeSize;

    int * perm = (int *) malloc(n * sizeof(int));
    thrust::sequence(perm, perm + n);
    thrust::default_random_engine g;
    thrust::shuffle(perm, perm + n, g);

	for(int i = 0; i < n; i++){
        int currentNode = perm[i];
		uint offset = str->cumDegs[currentNode];
		uint deg = str->deg(currentNode);

        //thrust::fill(usedColors, usedColors + n, false);

        for (uint j = 0; j < deg; j++) {
			uint neighID = str->neighs[offset + j];
			int jColor = col->coloring[neighID];
			if (jColor != -1 && jColor < deg) usedColors[offset + jColor] = true;
		}

        for(uint c = 0; c < deg; c++){
            if(!usedColors[offset + c]){
                col->coloring[currentNode] = c;
                break;
            }
        }
        
        if(col->coloring[currentNode] == -1) col->coloring[currentNode] = deg;
    }
}

Coloring* graphColoring(GraphStruct *str){
	int n = str->nodeSize;
    printf("%d ",n);

	Coloring* col = (Coloring * ) malloc(sizeof(Coloring));

    col->coloring = (int *) malloc(n * sizeof(int));
	thrust::fill(col->coloring, col->coloring + n, -1);

    bool * usedColors = (bool * ) malloc(str->edgeSize * sizeof(bool));
    thrust::fill(usedColors, usedColors + str->edgeSize, false);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start);

    CPUcolorer(col, str, usedColors);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo solo kernel %f ms\n", milliseconds);
    
	return col;
}