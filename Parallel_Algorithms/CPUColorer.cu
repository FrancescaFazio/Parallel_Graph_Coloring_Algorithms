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

void CPUcolorer(Coloring * col, GraphStruct *str){
	int n = str->nodeSize;

    int * perm = (int *) malloc(n * sizeof(int));
    thrust::sequence(perm, perm + n);
    thrust::default_random_engine g;
    thrust::shuffle(perm, perm + n, g);

	for(int i = 0; i < n; i++){
        int currentNode = perm[i];
		uint offset = str->cumDegs[currentNode];
		uint deg = str->deg(currentNode);

        for (uint j = 0; j < deg; j++) {
			uint neighID = str->neighs[offset + j];
			int jColor = col->coloring[neighID];
			if (jColor != -1) col->usedColors[jColor] = true;
		}

        for(uint c = 0; c < n; c++){
            if(!col->usedColors[c]) col->coloring[currentNode] = c;
        }
    }
}

Coloring* graphColoring(GraphStruct *str){
	int n = str->nodeSize;
    printf("%d ",n);

	Coloring* col = (Coloring * ) malloc(sizeof(Coloring));

    col->coloring = (int *) malloc(n * sizeof(int));
	thrust::fill(col->coloring, col->coloring + n, -1);

    col->usedColors = (bool * ) malloc(n * sizeof(bool));
    thrust::fill(col->usedColors, col->usedColors + n, false);

    CPUcolorer(col, str);

	return col;
}