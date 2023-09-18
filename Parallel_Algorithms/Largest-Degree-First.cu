#include <stdio.h>
#include <cuda.h>

//Libreria GNU C++: 
#include <bits/stdc++.h>

//Possiamo usare nomi per oggetti e variabili dalla libreria standard.
using namespace std;

//Costruzione del grafo.
void createGraph(int &n, int &m, int &maxDegree, int &minDegree, int** h_degree, int** h_adj, int** h_adj_p){
    
    //Prendo gli input dall'utente: numero di vertici, numero di archi e scelta di come costruire il grafo.
    int ch, i;
    cout << "Enter number of vertices\n";
    cin >> n;
    cout << "Enter number of edges\n";
    cin >> m;
    int k = (n * (n - 1)) / 2;
    if(k < m){
        cout << "Invalid number of edges...assigning graph to be complete graph" << endl;
        m = k;
    }
    cout << "Enter 0 for random and 1 for manual graph" << endl;
    cin >> ch;
    vector<set<int>> adj(n);

    //Genero il grafo in maniera casuale.
    if(ch == 1){       
        for(i = 0;i < m;i++)
        {
            cout << "Enter edge (0 is the first vertex)\n";
            int u,v;
            cin >> u >> v;
            if(adj[u].find(v) != adj[u].end()){
                cout << "Edge already present" << endl;
                i--;
                continue;
            }
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }

    //Genero il grafo in maniera manuale andando a definire tutti gli archi.
    else if(ch == 0){
        i = 0;
        while(i < m){
            int u = rand() % n;
            int v = rand() % n;
            if(adj[u].find(v) != adj[u].end())
                continue;
            if(u == v)
                continue;
            adj[u].insert(v);
            adj[v].insert(u);
            i++;
        }
    }

    *h_adj_p = new int[n + 1];
    *h_adj = new int[(2 * m) + 1];
    *h_degree = new int[n];

    int point = 0;
    for(int i = 0;i < n; i++){
        (*h_adj_p)[i] = point;
        for(auto j : adj[i])
            (*h_adj)[point++] = j;
    }

    (*h_adj_p)[n] = point;
    int mx = INT_MIN, mn = INT_MAX; 
    for(int i = 0;i < n; i++)
    {
        (*h_degree)[i] = (int)adj[i].size();
        mx = max(mx, (int)adj[i].size());
        mn = min(mn, (int)adj[i].size());
    } 
    minDegree = mn;
    maxDegree = mx;
    cout << "Max degree is : " << maxDegree << " and min degree is : " << minDegree << endl;
}

//Assegnamo i pesi in maniera randomica ai vertici.
void randomWeightAssign(int* h_weights, int n){                                             
    int i, j = 1;
    vector<int> arr(n);
    for(i = 0; i < n; i++)
        arr[i] = i;
    i = 0;
    while(i < n){
        int x = rand() % arr.size();
        h_weights[arr[x]] = j;
        j++;
        arr.erase(arr.begin() + x);
        i++;
    }
    return;
}

//Settiamo i vertici non colorati a true e il colore iniziale a invalid(0).
__global__ void initializeRemCol(bool* d_rem, int* d_colours, int n){               
    int ind = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(ind >= n)
        return;
    d_rem[ind] = true;
    d_colours[ind] = 0;
}

//Colorazione dei vertici.
__global__ void colorSet(int* d_adj, int* d_adj_p, int* d_weights, bool* d_rem, int* d_colours, int n, int maxDegree, int* d_degree){//pass h_deg here
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;    
    
    //Return se ho fatto scorrere tutti i vertici del grafo.
    if(index >= n)
        return;
    
    //Return se il vertice corrente è già colorato.
    if(!d_rem[index])                                                   
        return;

    //Inizializzazione variabili.
    int i, j, maxColours = maxDegree + 1;

    //Controlla se qualche vicino non colorato ha peso più alto.
    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){                      
        j = d_adj[i];                                                                   
        if(d_rem[j] && (d_degree[index]<d_degree[j]  || (d_degree[index]==d_degree[j] && d_weights[j] > d_weights[index])) )  
            return;
    }

    //Coloriamo il vertice corrente (ponendolo a false).
    d_rem[index] = false;                                              

    //Rimuoviamo il vertice dal set.
    bool* forbidden = new bool[maxColours + 1];
    for(i = 0; i < maxColours + 1; i++)
        forbidden[i] = false;

    //Trovo i colori assegnati al vicinato del vertice.
    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               
        j = d_adj[i];
        forbidden[d_colours[j]] = true;
    }

    //Assegno il colore che non è presente nei colori del vicinato.
    for(i = 1; i <= maxColours; i++){                                   
        if(!forbidden[i]){
            d_colours[index] = i;
            delete [] forbidden;
            return;
        }
    }
    delete [] forbidden;
}

//Colorazione del grafo
int* colourGraph(int n, int m, int maxDegree, int* d_adj, int* d_adj_p, int* d_weights, int* h_degree){
    
    //rem_size: numero vertici non colorati.
    int i, rem_size = n, mx = min(n, 512);   

    //h_rem: messi a true i vertici non colorati e a false i vertici colorati
    //h_colours: definisce i colori che vengono assegnati ai vertici.                                            
    bool* h_rem = new bool[n];
    bool* h_inc = new bool[n]; //?
    int* h_colours = new int[n];

    //Variabili h_rem e h_colours allocate nella GPU.
    bool* d_rem;
    int *d_colours;

    int *d_degree;

    //Allochiamo spazio sulla GPU.
    cudaMalloc((void**)&d_degree, sizeof(int) * n); 
    cudaMalloc((void**)&d_colours, sizeof(int) * n);                                        
    cudaMalloc((void**)&d_rem, sizeof(bool) * n);

    //Copiamo i dati sulla GPU.
    cudaMemcpy(d_degree, h_degree, sizeof(int) * n, cudaMemcpyHostToDevice);

    //Inizializzione
    initializeRemCol<<<mx, mx>>>(d_rem, d_colours, n);                                       

    while(rem_size > 0){
        // cout << rem_size << endl;

        //Lancio del kernel.
        colorSet<<<mx, mx>>>(d_adj, d_adj_p, d_weights, d_rem, d_colours, n, maxDegree, d_degree); 

        //Copia in memoria il set dei vertici non colorati aggiornato.
        cudaMemcpy(h_rem, d_rem, sizeof(bool) * n, cudaMemcpyDeviceToHost);                 
        int k = 0;
        for(i = 0; i < n; i++){
            if(h_rem[i])
                k++;
        }
        rem_size = k;
    }

    //Copio in memoria il set dei colori assegnati ai vertici.
    cudaMemcpy(h_colours, d_colours, sizeof(int) * n, cudaMemcpyDeviceToHost);

    //Libero la memoria.
    cudaFree(d_colours);                                                                    
    cudaFree(d_rem);
    cudaFree(d_degree);

    //Restituisco i colori assegnati ai vertici.
    return h_colours;
}

int main(){
    //n: numero dei vertici.
    //m: numero degli archi.
    //maxDegree: grado massimo.
    //minDegree: grado minimo.
    int n=0, m=0, maxDegree=0, minDegree=0;

    //Variabili che si riferiscono al vicinato dei vertici.
    int* h_adj = NULL, *h_adj_p = NULL;

    //Variabili h_adj e h_adj_p allocate per la GPU.
    int* d_adj, *d_adj_p;


    int* h_degree = NULL;

    srand(time(0)); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Creazione del grafo.
    createGraph(n, m, maxDegree, minDegree, &h_degree, &h_adj, &h_adj_p);
    
    //Alloco lo spazio sulla GPU.
    cudaMalloc((void**)&d_adj, sizeof(int) * ((2 * m) + 1));
    cudaMalloc((void**)&d_adj_p, sizeof(int) * (n + 1));
    
    //Copiamo i dati sulla GPU del vicinato.
    cudaMemcpy(d_adj, h_adj, sizeof(int) * ((2 * m) + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_p, h_adj_p, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);

    //h_weights: contine i pesi assegnati ai vertici.
    //d_weights: contine i pesi assegnati ai vertici sulla GPU.
    int* h_weights = new int[n];
    int* d_weights;

    //Alloco lo spazio sulla GPU.
    cudaMalloc((void**)&d_weights, sizeof(int) * n);

    //Assegnamento pesi in maniera randomica.
    randomWeightAssign(h_weights, n);

    //Copiamo i dati sulla GPU dei pesi.
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);

    // for(int i=0;i<n;i++)
    //     cout << h_weights[i] << endl;
    cudaEventRecord(start);

    //Colorazione del grafo
    int *colouring = colourGraph(n, m, maxDegree, d_adj, d_adj_p, d_weights, h_degree);  

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Stampo i colori assegnati ai nodi.
    // for(int i=0;i<n;i++)
    //     printf("Vertex %d : %d\n", i, colouring[i]);

    //Stampo in millisecondi quanto tempo ci ha messo a colorare il grafo.
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n",milliseconds);

    //Libero memoria
    cudaFree(d_adj);
    cudaFree(d_adj_p);
    cudaFree(d_weights);
}
