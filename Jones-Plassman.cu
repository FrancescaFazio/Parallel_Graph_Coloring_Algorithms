#include <stdio.h>
#include <cuda.h>

//Librerie GNU C++: 
#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

//Costruzione del grafo.
void readGraph(int &n, int &m, int &maxDegree, int** h_adj, int** h_adj_p){     

    //Prendo gli input dall'utente: numero di vertici, numero di archi e scelta di come costruire il grafo.
    int i, k, c;
    cout << "Enter the number of vertices : " << endl;
    cin >> n;
    cout << "Enter the number of edges : " << endl;
    cin >> m;
    if(m > ((n * (n - 1))/ 2)){
        cout << "Invalid number of edges." << endl;
        return;
    }

    cout << "Random graph or manual? Enter 0 or 1" << endl;
    cin >> c;

    vector<set<int>> adj(n);

    //Genero il grafo in maniera casuale.
    if(c == 0){                                                         
        i = 0;
        while(i < m){
            int x, y;
            do{
                x = rand() % n;
                y = rand() % n;
            }while(x == y);

            if(adj[x].find(y) != adj[x].end())
                continue;
            // printf("%d --- %d\n", x, y);
            adj[x].insert(y);
            adj[y].insert(x);
            i++;
        }
    }

    //Genero il grafo in maniera automatica andando a definire tutti gli archi.
    else{                                                              
        i = 0;
        while(i < m){
            printf("Click 1 to enter edge and 0 to finish.\n");
            scanf("%d", &k);
            if(!k)
                break;
            int s, d;
            printf("Enter start and end of edge in 1-ordering : \n");
            scanf("%d %d", &s, &d);
            if(s == d){
                printf("Invalid edge.\n");
                continue;
            }
            if(s > n || s < 1 || d > n || d < 1){
                printf("Invalid edge.\n");
                continue;
            }
            adj[s - 1].insert(d - 1);
            adj[d - 1].insert(s - 1);
            i++;
        }
    }
    *h_adj_p = new int[n + 1];
    *h_adj = new int[(2 * m) + 1];

    int point = 0;
    for(i = 0;i < n; i++){
        (*h_adj_p)[i] = point;
        for(auto j : adj[i])
            (*h_adj)[point++] = j;
    }
    (*h_adj_p)[n] = point;

    //Calcolo il grado massimo del grafo generato.
    int mx = INT_MIN; 
    for(i = 0;i < n; i++)                                                           
        mx = max(mx, (int)adj[i].size());
    
    maxDegree = mx;
}

//Assegnamento random dei pesi ai vertici.
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

//Colorazione del grafo.
int* colourGraph(int n, int m, int maxDegree, int* d_adj, int* d_adj_p, int* d_weights){
    
    //rem_size: numero vertici non colorati.
    int i, rem_size = n, mx = min(n, 512); 

    //h_rem: messi a true i vertici non colorati e a false i vertici colorati
    //h_colours: definisce i colori che vengono assegnati ai vertici.
    bool* h_rem = new bool[n];
    int* h_colours = new int[n];
    
    //Variabili h_rem e h_colours allocate nella GPU.
    bool* d_rem;
    int *d_colours;

    //Allochiamo spazio sulla GPU.
    cudaMalloc((void**)&d_colours, sizeof(int) * n);                                        
    cudaMalloc((void**)&d_rem, sizeof(bool) * n);

    //Inizializzione
    initializeRemCol<<<mx, mx>>>(d_rem, d_colours, n);                                       

    while(rem_size > 0){
        // cout << rem_size << endl;

        //Lancio del kernel.
        colorSet<<<mx, mx>>>(d_adj, d_adj_p, d_weights, d_rem, d_colours, n, maxDegree);    

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

    //Restituisco i colori assegnati ai vertici.
    return h_colours;
}

//Inizializzazione vertici non colorati a true e il colore iniziale a invalid(0)
__global__ void initializeRemCol(bool* d_rem, int* d_colours, int n){              
    int ind = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(ind >= n)
        return;
    d_rem[ind] = true;
    d_colours[ind] = 0;
}

//Colorazione dei vertici.
__global__ void colorSet(int* d_adj, int* d_adj_p, int* d_weights, bool* d_rem, int* d_colours, int n, int maxDegree){
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
        if(d_rem[j] && d_weights[j] > d_weights[index])
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
            // printf("%d : %d\n", index, i);
            delete [] forbidden;
            return;
        }
    }
    delete [] forbidden;
}

int main(){
    //n: numero vertici.
    //m: numero archi.
    //maxDegree: massimo grado del grafo.
    int n, m, maxDegree, i; 
    srand(time(0)); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //h_adj: 
    //h_adj_p: 
    int* h_adj = NULL, *h_adj_p = NULL;

    //Variabili h_adj e h_adj_p sulla GPU.
    int* d_adj, *d_adj_p;

    //Vado a costruire il grafo.
    readGraph(n, m, maxDegree, &h_adj, &h_adj_p);

    //h_weights: contine i pesi assegnati ai vertici.
    //d_weights: contine i pesi assegnati ai vertici sulla GPU.
    int* h_weights = new int[n];
    int* d_weights;

    //Alloco lo spazio sulla GPU.
    cudaMalloc((void**)&d_adj, sizeof(int) * ((2 * m) + 1));                            
    cudaMalloc((void**)&d_adj_p, sizeof(int) * (n + 1));
    cudaMalloc((void**)&d_weights, sizeof(int) * n);

    //Assegnamento dei pesi.
    randomWeightAssign(h_weights, n);

    //Copiamo i dati sulla GPU del vicinato e dei pesi.
    cudaMemcpy(d_adj, h_adj, sizeof(int) * ((2 * m) + 1), cudaMemcpyHostToDevice);      
    cudaMemcpy(d_adj_p, h_adj_p, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);

    //Visualizzazione a schermo del grado massimo del grafo.
    cout << "The max degree is : " << maxDegree << endl;

    // for(i = 0;i < n; i++)
    //     cout << "Node " << i << " : " << h_weights[i] << endl;
    cudaEventRecord(start);
    
    //Colorazione del grafo.
    int *colouring = colourGraph(n, m, maxDegree, d_adj, d_adj_p, d_weights);        
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Stampo i colori assegnati ai nodi.
    for(i = 0;i < n; i++)
        cout << "Colour of node " << i << " is : " << colouring[i] << endl;

    //Stampo in millisecondi quanto tempo ci ha messo a colorare il grafo.
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    //Libero la memoria.
    cudaFree(d_adj_p);
    cudaFree(d_adj);    
    cudaFree(d_weights);
}