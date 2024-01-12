#include <iostream>
#include "graph\graph.h"
#include "graph\coloring.h"
#include "utils\common.h"

#define THREADxBLOCK 128

int main(void) {
    //unsigned int n = 50515;		
    //float prob = 0;				    
    //std::default_random_engine eng{0}; 

    srand(time(0));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // new graph with n nodes
    Graph graph("facebook_clean_data/com-amazon.ungraph.txt", 1);
    //Graph graph(n, 1);
    //graph.randGraph(prob, eng);

    // get the graph struct
    GraphStruct *str = graph.getStruct();

    cudaEventRecord(start);

    Coloring* col = graphColoring(str);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Stampo in millisecondi quanto tempo ci ha messo a colorare il grafo.
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    int maxColor = 0;
    for(int i = 0; i < str->nodeSize; i++){
        if(maxColor < col->coloring[i]) maxColor = col->coloring[i];
        //printf("%d ", col->coloring[i]);
    }
    printf("\nColore massimo: %d", maxColor+1);
    //printColoring(col, str, 1);

    return EXIT_SUCCESS;
}