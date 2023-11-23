#include <iostream>
#include "graph\graph.h"
#include "graph\coloring.h"
#include "utils\common.h"

#define THREADxBLOCK 128

// da fare: aggiungere stream? ad esempio per eseguire in modo concorrente stampa del grafo e la sua colorazione

int main(void) {
    unsigned int n = 5000;		 // number of nodes for random graphs
    float prob = 1;				    // density (percentage) for random graphs
    std::default_random_engine eng{0};  // fixed seed

    srand(time(0));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // new graph with n nodes
    Graph graph(n, 1);

    // generate a random graph
    graph.randGraph(prob, eng);

    // get the graph struct
    GraphStruct *str = graph.getStruct();

    //print_d <<< 1, 1 >>> (str, true);
    cudaDeviceSynchronize();

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
    printf("Coloratura trovata: ");
    for(int i = 0; i < str->nodeSize; i++){
        if(maxColor < col->coloring[i]) maxColor = col->coloring[i];
        printf("%d ", col->coloring[i]);
    }
    printf("\nColore massimo: %d", maxColor+1);
    //printColoring(col, str, 1);

    return EXIT_SUCCESS;
}