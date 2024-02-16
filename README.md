Per eseguire il codice su grafo amazon,

nvcc -dc -I . .\Parallel_Algorithms\testerColorer.cu .\Parallel_Algorithms\[Algoritmo]Colorer.cu .\Parallel_Algorithms\graph\graph_d.cu .\Parallel_Algorithms\graph\graph.cpp

nvcc testerColorer.obj [Algoritmo]Colorer.obj graph.obj graph_d.obj -o tester

.\tester.exe 548551 data\com-amazon.ungraph.txt
