#include <iostream>
#include "graph\graph.h"
#include "graph\coloring.h"
#include "utils\common.h"

#define THREADxBLOCK 128

int main(int argc,  char **argv) {
    unsigned int n;
    sscanf (argv[1],"%d",&n);
    char * fpath = argv[2];
    for (int i = 0; i < argc; ++i)

    //unsigned int n = 50000;		
    //float prob = 0.0001;				    
    //sstd::default_random_engine eng{0}; 

    srand(time(0));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Graph graph(fpath, n, 1);
    //Graph graph(n, 1);
    //graph.randGraph(prob, eng);

    GraphStruct *str = graph.getStruct();

    cudaEventRecord(start);

    Coloring* col = graphColoring(str);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    //print_d<<<1,1>>>(str, 1);
    cudaDeviceSynchronize();

    int maxColor = 0;
    for(int i = 0; i < str->nodeSize; i++){
        if(maxColor < col->coloring[i]) maxColor = col->coloring[i];
    }
    printf("\nColore massimo: %d", maxColor);

    return EXIT_SUCCESS;
}