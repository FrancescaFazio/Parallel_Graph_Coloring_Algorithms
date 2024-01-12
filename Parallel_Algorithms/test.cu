#include <iostream>
#include "graph\graph.h"
#include "graph\coloring.h"
#include "utils\common.h"


int testProb0(){
    int n = 1000;		 
    std::default_random_engine eng{0}; 
    
    Graph graph(n, 1);
    graph.randGraph(0, eng);

    GraphStruct *str = graph.getStruct();

    Coloring* col = graphColoring(str);
    cudaDeviceSynchronize();

    int maxColor = 0;
    for(int i = 0; i < str->nodeSize; i++){
        if(maxColor < col->coloring[i]) maxColor = col->coloring[i];
    }

    if (maxColor == 1) return 1;
    else return 0;
}

int testProb1(){
    int n = 1000;		 
    std::default_random_engine eng{0}; 
    
    Graph graph(n, 1);
    graph.randGraph(0, eng);

    GraphStruct *str = graph.getStruct();

    Coloring* col = graphColoring(str);
    cudaDeviceSynchronize();

    int maxColor = 0;
    for(int i = 0; i < str->nodeSize; i++){
        if(maxColor < col->coloring[i]) maxColor = col->coloring[i];
    }

    if (maxColor == 1) return 1;
    else return 0;
}

int testGraph1(){
    return 1;
}

int testGraph2(){
    return 1;
}