#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include "graph.h"

using namespace std;

int countLines(const char * fp){
	FILE * file = fopen(fp, "r");
	int lines = 0;
	if ( file == NULL ) return -1;
	while (EOF != (fscanf(file, "%*[^\n]"), fscanf(file,"%*c"))) ++lines;
	fclose(file);

	return lines;
}

/**
 * Generate an Erdos random graph
 * @param n number of nodes
 * @param density probability of an edge (expected density)
 * @param eng seed
 */
void Graph::setup(node_sz nn) {
	if (GPUEnabled){
		memsetGPU(nn, string("nodes"));
		memsetGPU(nn, string("weights"));
	}
	else {
		str = new GraphStruct();
		str->cumDegs = new node[nn + 1]{};  // starts by zero
		str->weights = new int[nn]{};
	}
	str->nodeSize = nn;
}

/**
 * Crea un grafo leggendo da file .csv in formato lista d'adiacenza.
*/
Graph::Graph(const char * filePath, bool GPUEnb){
	GPUEnabled = GPUEnb;
	int n = 4997962;
	FILE * file = fopen(filePath, "r");
	setup(n);

	vector<int>* edges = new vector<int>[n];
	int i,j;

	while (fscanf(file, "%d\t%d", &i, &j) == 2) {
		//printf("%d %d\n", i,j);
		edges[i].push_back(j);
		edges[j].push_back(i);
		str->cumDegs[i + 1]++;
		str->cumDegs[j + 1]++;
		str->edgeSize += 2;
    }

	fclose(file);

	for (int i = 0; i < n; i++)
		str->cumDegs[i + 1] += str->cumDegs[i];

	maxDeg = 0;
	minDeg = n;
	for (int i = 0; i < n; i++) {
		if (str->deg(i) > maxDeg)
			maxDeg = str->deg(i);
		if (str->deg(i) < minDeg)
			minDeg = str->deg(i);
	}
	density = (float) str->edgeSize / (float) (n * (n - 1));
	meanDeg = (float) str->edgeSize / (float) n;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	// manage memory for edges with CUDA Unified Memory
	if (GPUEnabled)
		memsetGPU(n,"edges");
	else
		str->neighs = new node[str->edgeSize] { };

	for (int i = 0; i < n; i++)
		memcpy((str->neighs + str->cumDegs[i]), edges[i].data(), sizeof(int) * edges[i].size());
}

/**
 * Generate a new random graph
 * @param eng seed
 */
void Graph::randGraph(float prob, std::default_random_engine & eng) {
	if (prob < 0 || prob > 1) {
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n");
	}
	uniform_real_distribution<> randR(0.0, 1.0);
	node n = str->nodeSize;

	// gen edges
	vector<int>* edges = new vector<int>[n];
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++)
			if (randR(eng) < prob) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				str->cumDegs[i + 1]++;
				str->cumDegs[j + 1]++;
				str->edgeSize += 2;
			}
	}
	for (int i = 0; i < n; i++)
		str->cumDegs[i + 1] += str->cumDegs[i];

	// max, min, mean deg
	maxDeg = 0;
	minDeg = n;
	for (int i = 0; i < n; i++) {
		if (str->deg(i) > maxDeg)
			maxDeg = str->deg(i);
		if (str->deg(i) < minDeg)
			minDeg = str->deg(i);
	}
	density = (float) str->edgeSize / (float) (n * (n - 1));
	meanDeg = (float) str->edgeSize / (float) n;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	// manage memory for edges with CUDA Unified Memory
	if (GPUEnabled)
		memsetGPU(n,"edges");
	else
		str->neighs = new node[str->edgeSize] { };

	for (int i = 0; i < n; i++)
		memcpy((str->neighs + str->cumDegs[i]), edges[i].data(), sizeof(int) * edges[i].size());
}

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void Graph::print(bool verbose) {
	node n = str->nodeSize;
	cout << "** Graph (num node: " << n << ", num edges: " << str->edgeSize
			<< ")" << endl;
	cout << "         (min deg: " << minDeg << ", max deg: " << maxDeg
		 << ", mean deg: " << meanDeg << ", connected: " << connected << ")"
		 << endl;

	if (verbose) {
		for (int i = 0; i < n; i++) {
			cout << "   node(" << i << ")" << "["
					<< str->cumDegs[i + 1] - str->cumDegs[i] << "]-> ";
			for (int j = 0; j < str->cumDegs[i + 1] - str->cumDegs[i]; j++) {
				cout << str->neighs[str->cumDegs[i] + j] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}