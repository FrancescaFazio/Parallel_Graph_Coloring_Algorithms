#include "graph\coloring.h"

int main(void) {
	unsigned int n = 1000;		 // number of nodes for random graphs
	float prob = 0.5;				    // density (percentage) for random graphs
	std::default_random_engine eng{0};  // fixed seed

	srand(time(0));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// new graph with n nodes
	Graph graph(n,1);

	// generate a random graph
	graph.randGraph(prob,eng);

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

	uint maxColor = col->coloring[0];
 	printf("Coloratura trovata: ");
	for(int i = 0; i < str->nodeSize; i++){
			if(maxColor < col->coloring[i]) maxColor = col->coloring[i];
			printf("%d ", col->coloring[i]);
	}
	printf("\nColore massimo: %d", maxColor+1);

	return EXIT_SUCCESS;
}