/**
 * @file mpi_bellman_ford.cpp
 * @author Soham Tripathy (https://archaic-mage.vercel.app)
 * @brief Finding the shortest path using Bellman-Ford algorithm in a distributed manner using MPI
 * @version 0.1
 * @date 2023-10-16
 *
 * @ref https://snap.stanford.edu/data/
 * @ref https://www.mcs.anl.gov/research/projects/mpi/
 * 
 * Compile: mpic++ -std=c++11 -o mpi_bellman_ford mpi_bellman_ford.cpp
 * Run: mpiexec -n <number of processes> ./mpi_bellman_ford <input file>, you will find the output file 'output.txt'
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <limits.h>
#include <unistd.h>
#include <cstdarg>

using namespace std;

#define INF 1000000

/**
 * @brief Print the debug messages
 *
 * @param rank      the rank of the current process
 * @param format    the format of the message
 * @param ...       the arguments
 */
void debug(int rank, char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("Rank %d: ", rank);
    vprintf(format, args);
    fflush(stdout);
    va_end(args);
}

/**
 * @brief Translate 2-dimension coordinate to 1-dimension
 *
 * @param i     the row index
 * @param j     the column index
 * @param n     the number of columns
 * @return int  the 1-dimension index
 */
int access2d(int i, int j, int n) {
    return i * n + j;
}

/**
 * @brief Read the graph from the file
 *
 * @param filename  the name of the file
 * @param n         the number of vertices
 * @param graph     the adjacency matrix of the graph
 */
void read_graph(string filename, int &n, int* &graph)
{
    ifstream file(filename);
    if (file.is_open())
    {
        file >> n;
        graph = (int*) malloc(sizeof(int) * n * n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                file >> graph[access2d(i, j, n)];
            }
        }
    }
    else
    {
        cerr << "FILE NOT FOUND" << endl;
    }
}

/**
 * @brief Print the result in a file named 'output.txt'
 *
 * @param dist  the distance vector
 * @param N     the number of vertices
 */
void print_result(int N, int* dist) {
    ofstream file("output.txt");
    if(file.is_open()) {
        for(int i = 0; i < N; i++) {
            file << dist[i] << endl;
        }
    } else {
        cerr << "FILE NOT FOUND" << endl;
    }
}

/**
 * @brief Bellman Ford algorithm to find the shortest path. It is a distributed algorithm to find the shortest path from node 0 to all other nodes.
 *
 * @param rank              the rank of the current process
 * @param np                the number of processes
 * @param comm              the communicator
 * @param n                 the number of vertices
 * @param graph             the adjacency matrix of the graph
 * @param dist              the distance vector
 * @param has_neg_cycle     to check if there is a negative cycle
 */
void bellman_ford(int rank, int np, MPI_Comm comm, int n, int* graph, int* dist, bool *has_neg_cycle)
{

    int local_n;                     // local copy of the number of vertices
    int local_start, local_end;      // local working range
    int* local_graph; // local copy of the graph
    int* local_dist;          // local copy of the distance vector

    // get the local number of vertices
    if (rank == 0)
    {
        local_n = n;
    }
    // Broadcast the local number of vertices to all processes
    MPI_Bcast(&local_n, 1, MPI_INT, 0, comm);

    // compute the local working range
    local_start = rank * (local_n / np);
    local_end = (rank + 1) * (local_n / np); // till next process start
    // last process will take care of the remaining vertices
    if (rank == np - 1)
    {
        local_end = local_n;
    }
    MPI_Barrier(comm);

    local_graph = (int*) malloc(sizeof(int) * local_n * local_n);
    local_dist = (int*) malloc(sizeof(int) * local_n);

    // initialize the local graph and distance vector
    if (rank == 0)
    {
        memcpy(local_graph, graph, sizeof(int) * local_n * local_n);
    }
    // broadcast the local graph and distance vector to all processes
    MPI_Bcast(local_graph, local_n * local_n, MPI_INT, 0, comm);

    // compute the shortest path
    for (int i = 0; i < local_n; i++)
    {
        local_dist[i] = INF;
    }

    local_dist[0] = 0;
    // wait for all processes to have the local graph and distance vector
    MPI_Barrier(comm);

    // relax the edges
    bool is_changed;
    int iter;
    for (iter = 0; iter < local_n - 1; iter++) {
        is_changed = false;
        for (int u = local_start; u < local_end; u++) {
            for (int v = 0; v < local_n; v++) {
                int weight = local_graph[u * local_n + v];
                if (weight < INF) {
                    if (local_dist[u] + weight < local_dist[v]) {
                        local_dist[v] = local_dist[u] + weight;
                        is_changed = true;
                    }
                }
            }
        }
        debug(rank, "iter = %d\n", iter);
        // wait for all processes to complete the iteration and check if there is a change
        MPI_Allreduce(MPI_IN_PLACE, &is_changed, 1, MPI_CXX_BOOL, MPI_LOR, comm);
        debug(rank, "is_changed = %d\n", is_changed);
        if(!is_changed) {
            break;
        }
        for(int i = 0; i<n; i++) {
            debug(rank, "dist[%d] = %d\n", i, local_dist[i]);
        }
        // sync the distance vector
        MPI_Allreduce(MPI_IN_PLACE, local_dist, local_n, MPI_INT, MPI_MIN, comm);
    }

    if(iter == local_n - 1) {
        //we do one more iteration to check if there is a negative cycle
        is_changed = false;
        for (int u = local_start; u < local_end; u++) {
            for (int v = 0; v < local_n; v++) {
                int weight = local_graph[u * local_n + v];
                if (weight < INF) {
                    if (local_dist[u] + weight < local_dist[v]) {
                        local_dist[v] = local_dist[u] + weight;
                        is_changed = true;
                    }
                }
            }
        }
        // combines the results of all processes using the logical OR operation and stores the result in has_neg_cycle
        MPI_Allreduce(&is_changed, has_neg_cycle, 1, MPI_CXX_BOOL, MPI_LOR, comm);
    }


    // copy the local distance vector to the global distance vector
    if(rank == 0) memcpy(dist, local_dist, local_n * sizeof(int));

    // free the memory
    free(local_graph);
    free(local_dist);
}

int main(int argc, char **argv)
{

    if (argc < 1)
    {
        cerr << "INPUT FILE WAS NOT FOUND" << endl;
    }

    string filename = argv[1];
    bool has_neg_cycle = false;
    int *graph;
    int *dist;
    int N;

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // get the rank and the number of processes
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    // read the graph from the file in the root process
    if(rank == 0) {
        read_graph(filename, N, graph);
        dist = new int[N];
    }

    double start_time, end_time;
    MPI_Barrier(comm);
    start_time = MPI_Wtime();

    // run the bellman ford algorithm
    bellman_ford(rank, np, comm, N, graph, dist, &has_neg_cycle);
    MPI_Barrier(comm);

    // end the timer
    end_time = MPI_Wtime();

    if(rank == 0) {
        cout << "Time taken: " << end_time - start_time << endl;
        if(has_neg_cycle) {
            cout << "The graph has a negative cycle" << endl;
        } else {
            print_result(N, dist);
        }
        free(graph);
        free(dist);
    }

    // finalize MPI
    MPI_Finalize();

    return 0;
}