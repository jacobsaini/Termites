#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "termites.h"

#define B       20    // num of termites
#define T       100000 // num of steps
#define N       20     // nxn grid size
#define D       20      // density (%) of chips compared to grid

struct termite {
    int has_chip;
    int moves;
    int idx;
};

struct spot {
    int chip;
    int has_t;
};

__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < B) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void go(struct spot* grid, struct termite* termites, curandState* state) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < B) {
	if (termites[s].moves == 1) {
	    return;
	}
        if (termites[s].has_chip == 1) {
            find_new_pile(s, grid, termites, state);
        } else {
            search_for_chip(s, grid, termites, state);
        }
    }

    __syncthreads();
}
__device__ void update_grid_pos(int s, int new_pos, struct spot* grid, struct termite* termites) {
    int old_pos = termites[s].idx;
    atomicExch(&grid[old_pos].has_t, 0);
    termites[s].has_chip = 1;
    termites[s].moves = 1;
    termites[s].idx = new_pos;
    atomicExch(&grid[new_pos].chip, 0);
    atomicExch(&grid[new_pos].has_t, 1);
}


__device__ void search_for_chip(int i, struct spot* grid, struct termite* termites, curandState* state) {
    int t_pos = termites[i].idx;
    if ((t_pos - N) >= 0 && grid[t_pos - N].chip == 1) {
        update_grid_pos(i, t_pos - N, grid, termites);
        return;
    } else if ((t_pos - 1) % N != (N - 1) && t_pos - 1 > 0 && grid[t_pos - 1].chip == 1) {
        update_grid_pos(i, t_pos - 1, grid, termites);
        return;
    } else if ((t_pos + N) < (N * N) && grid[t_pos + N].chip == 1) {
        update_grid_pos(i, t_pos + N, grid, termites);
        return;
    } else if ((t_pos + 1) % N != 0 && grid[t_pos + 1].chip == 1) {
        update_grid_pos(i, t_pos + 1, grid, termites);
        return;
    } else {
        wiggle(i, grid, termites, state);
        if (termites[i].idx == t_pos) {
            get_away(i, grid, termites, state);
        }
        search_for_chip(i, grid, termites, state);
    }
}

__device__ void find_new_pile(int i, struct spot* grid, struct termite* termites, curandState* state) {
    int t_pos = termites[i].idx;
    if ((t_pos - N) >= 0 && grid[t_pos - N].chip == 1) {
        put_down_chip(i, grid, termites, state);
    } else if ((t_pos - 1) % N != (N - 1) && t_pos - 1 > 0 && grid[t_pos - 1].chip == 1) {
        put_down_chip(i, grid, termites, state);
    } else if ((t_pos + N) < (N * N) && grid[t_pos + N].chip == 1) {
        put_down_chip(i, grid, termites, state);
    } else if ((t_pos + 1) % N != 0 && grid[t_pos + 1].chip == 1) {
        put_down_chip(i, grid, termites, state);
    } else {
        wiggle(i, grid, termites, state);
        if (termites[i].idx == t_pos) {
            get_away(i, grid, termites, state);
        }
        find_new_pile(i, grid, termites, state);
    }
}

__device__ void put_down_chip(int i, struct spot* grid, struct termite* termites, curandState* state) {
    int old_t_pos = termites[i].idx;
    atomicExch(&grid[old_t_pos].chip, 1);
    termites[i].has_chip = 0;
    termites[i].moves = 1;
    get_away(i, grid, termites, state);
}

__device__ void get_away(int i, struct spot* grid, struct termite* termites, curandState* state) {
    int new_pos = curand(&state[i]) % (N * N);
    while (atomicCAS(&grid[new_pos].has_t, 0, 1) != 0 || grid[new_pos].chip == 1) {
        new_pos = curand(&state[i]) % (N * N);
    }
    int old_t_pos = termites[i].idx;
    atomicExch(&grid[old_t_pos].has_t, 0);
    termites[i].idx = new_pos;
}

__device__ void wiggle(int i, struct spot* grid, struct termite* termites, curandState* state) {
    int arr[4] = {0, 1, 2, 3};
    int new_pos;
    int old_pos = termites[i].idx;
    int dir;
    for (int c = 0; c < 4; c++) {
        dir = arr[curand(&state[i]) % 4];
        switch (dir) {
            case 0:
                new_pos = old_pos - N;
                if (new_pos >= 0 && (grid[new_pos].chip != 1 && atomicCAS(&grid[new_pos].has_t, 0, 1) == 0)) {
                    atomicExch(&grid[old_pos].has_t, 0);
                    termites[i].idx = new_pos;
                    return;
                }
                break;
            case 1:
                new_pos = old_pos - 1;
                if ((new_pos % N != (N - 1)) && new_pos > 0 && (grid[new_pos].chip != 1 && atomicCAS(&grid[new_pos].has_t, 0, 1) == 0)) {
                    atomicExch(&grid[old_pos].has_t, 0);
                    termites[i].idx = new_pos;
                    return;
                }
                break;
            case 2:
                new_pos = old_pos + N;
                if (new_pos < (N * N) && (grid[new_pos].chip != 1 && atomicCAS(&grid[new_pos].has_t, 0, 1) == 0)) {
                    atomicExch(&grid[old_pos].has_t, 0);
                    termites[i].idx = new_pos;
                    return;
                }
                break;
            case 3:
                new_pos = old_pos + 1;
                if ((new_pos % N != 0) && (grid[new_pos].chip != 1 && atomicCAS(&grid[new_pos].has_t, 0, 1) == 0)) {
                    atomicExch(&grid[old_pos].has_t, 0);
                    termites[i].idx = new_pos;
                    return;
                }
                break;
        }
    }
}

void setup(struct spot* grid, struct termite* termites) {
    srand(time(NULL));

    // Init grid values
    int grid_size = N * N;
    for (int i = 0; i < grid_size; i++) {
        grid[i].chip = 0;
        grid[i].has_t = 0;
    }

    // Randomly place chips
    int chip_amount = (grid_size) * (D * 0.01);
    int i = rand() % (grid_size);
    while (chip_amount > 0) {
        i = rand() % (grid_size);
        if (grid[i].chip != 1 && chip_amount != 0) {
            grid[i].chip = 1;
            chip_amount = chip_amount - 1;
        }
    }

    // Randomly place termites
    for (int b = 0; b < B; b++) {
        int i = rand() % (grid_size);
        while ((grid[i].chip != 0) && (grid[i].has_t != 0)) {
            i = rand() % (grid_size);
        }
        termites[b].has_chip = 0;
        termites[b].moves = 0;
        termites[b].idx = i;
        grid[i].has_t = 1;
    }
}

void print_progress(struct spot* grid, int step) {
    printf("Step: %d\n", step);
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            if (grid[k + (j * N)].has_t == 1) {
                printf(" T ");
            } else if (grid[k + (j * N)].chip) {
	        printf(" C ");
	    } else {
		printf(" . ");
            }
        }
        printf("\n");
    }
}

__global__ void reset_moves(struct termite* termites) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < B) {
        termites[s].moves = 0;
    }
}

int main() {
    struct spot* d_grid;
    struct termite* d_termites;
    struct spot* h_grid = (struct spot*)malloc(N * N * sizeof(struct spot));
    struct termite* h_termites = (struct termite*)malloc(B * sizeof(struct termite));
    // For random number stuff 
    curandState* d_state;

    setup(h_grid, h_termites);

    cudaMalloc(&d_grid, N * N * sizeof(struct spot));
    cudaMalloc(&d_termites, B * sizeof(struct termite));
    cudaMalloc(&d_state, B * sizeof(curandState));

    int threadsPerBlock = 256;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;

    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, time(NULL));
    cudaDeviceSynchronize();
    
    print_progress(h_grid, 0);
    for (int i = 0; i < T; i++) {
        cudaMemcpy(d_grid, h_grid, N * N * sizeof(struct spot), cudaMemcpyHostToDevice);
        cudaMemcpy(d_termites, h_termites, B * sizeof(struct termite), cudaMemcpyHostToDevice);

        go<<<blocksPerGrid, threadsPerBlock>>>(d_grid, d_termites, d_state);
        cudaDeviceSynchronize();

        reset_moves<<<blocksPerGrid, threadsPerBlock>>>(d_termites);
        cudaDeviceSynchronize();

        cudaMemcpy(h_grid, d_grid, N * N * sizeof(struct spot), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_termites, d_termites, B * sizeof(struct termite), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(h_grid, d_grid, N * N * sizeof(struct spot), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_termites, d_termites, B * sizeof(struct termite), cudaMemcpyDeviceToHost);

    print_progress(h_grid, T);
    
    cudaFree(d_grid);
    cudaFree(d_termites);
    cudaFree(d_state);
    free(h_grid);
    free(h_termites);


    return 0;
}


