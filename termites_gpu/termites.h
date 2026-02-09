// termites_gpu.h

#ifndef TERMITES_H
#define TERMITES_H

#include <curand_kernel.h>

// CUDA kernel declarations
__global__ void setup_kernel(curandState* state, unsigned long seed);
__global__ void go(struct spot* grid, struct termite* termites, curandState* state);

// Device function declarations
__device__ void update_grid_pos(int s, int new_pos, struct spot* grid, struct termite* termites);
__device__ void search_for_chip(int i, struct spot* grid, struct termite* termites, curandState* state);
__device__ void find_new_pile(int i, struct spot* grid, struct termite* termites, curandState* state);
__device__ void put_down_chip(int i, struct spot* grid, struct termite* termites, curandState* state);
__device__ void get_away(int i, struct spot* grid, struct termite* termites, curandState* state);
__device__ void wiggle(int i, struct spot* grid, struct termite* termites, curandState* state);

#endif // TERMITES_GPU_H

