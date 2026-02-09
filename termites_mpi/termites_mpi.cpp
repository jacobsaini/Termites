#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "termites.h"

#define B 1000    // num of termites
#define T 1000000     // num of steps
#define N 500     // nxn grid size
#define D 20      // density (%) of chips compared to grid

struct termite {
    int has_chip;
    int moves;
    int idx;
};

struct spot {
    int chip;
    int has_t;
};

// Global variables to store total times
double setup_time_total = 0.0;
double go_time_total = 0.0;
double search_for_chip_time_total = 0.0;
double find_new_pile_time_total = 0.0;
double put_down_chip_time_total = 0.0;
double get_away_time_total = 0.0;
double wiggle_time_total = 0.0;

void print_progress(struct spot* grid, int step) {
    printf("Step: %d", step);
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

void setup(struct spot grid[], struct termite termites[], int rank) {
    double start_time = MPI_Wtime();

    srand(time(NULL) + rank); 

    int grid_size = N * N;
    for (int i = 0; i < grid_size; i++) {
        grid[i].chip = 0;
        grid[i].has_t = 0;
    }

    int chip_amount = (grid_size) * (D * 0.01);
    while (chip_amount > 0) {
        int i = rand() % (grid_size);
        if (grid[i].chip != 1) {
            grid[i].chip = 1;
            chip_amount--;
        }
    }

    for (int b = 0; b < B; b++) {
        int i;
        do {
            i = rand() % (grid_size);
        } while (grid[i].chip != 0 || grid[i].has_t != 0);
        termites[b].has_chip = 0;
        termites[b].moves = 0;
        termites[b].idx = i;
        grid[i].has_t = 1;
    }

    double end_time = MPI_Wtime();
    setup_time_total += end_time - start_time;
}

void go(struct spot grid[], struct termite termites[], int rank, int size) {
    double start_time = MPI_Wtime();

    int start = rank * (B / size);
    int end = (rank + 1) * (B / size);

    for (int s = start; s < end; s++) {
        if (termites[s].moves == 1) continue;
        if (termites[s].has_chip == 1) {
            find_new_pile(s, grid, termites, rank);
        } else {
            search_for_chip(s, grid, termites, rank);
        }
    }

    for (int m = start; m < end; m++) {
        termites[m].moves = 0;
    }

    double end_time = MPI_Wtime();
    go_time_total += end_time - start_time;
}

void update_grid_pos(int s, int new_pos, struct spot grid[], struct termite termites[]) {
    int old_pos = termites[s].idx;
    grid[old_pos].has_t = 0;

    termites[s].has_chip = 1;
    termites[s].moves = 1;
    termites[s].idx = new_pos;

    grid[new_pos].chip = 0;
    grid[new_pos].has_t = 1;
}

void search_for_chip(int i, struct spot grid[], struct termite termites[], int rank) {
    double start_time = MPI_Wtime();

    int t_pos = termites[i].idx;
    if ((t_pos - N) >= 0 && grid[t_pos - N].chip == 1) {
        update_grid_pos(i, t_pos - N, grid, termites);
        goto end;
    }
    if ((t_pos - 1) % N != (N - 1) && t_pos - 1 >= 0 && grid[t_pos - 1].chip == 1) {
        update_grid_pos(i, t_pos - 1, grid, termites);
        goto end;
    }
    if ((t_pos + N) < (N * N) && grid[t_pos + N].chip == 1) {
        update_grid_pos(i, t_pos + N, grid, termites);
        goto end;
    }
    if ((t_pos + 1) % N != 0 && grid[t_pos + 1].chip == 1) {
        update_grid_pos(i, t_pos + 1, grid, termites);
        goto end;
    }

    wiggle(i, grid, termites, rank);
    if (termites[i].idx == t_pos) {
        get_away(i, grid, termites, rank);
    }
    search_for_chip(i, grid, termites, rank);
    
end:
    double end_time = MPI_Wtime();
    search_for_chip_time_total += end_time - start_time;
}

void find_new_pile(int i, struct spot grid[], struct termite termites[], int rank) {
    double start_time = MPI_Wtime();

    int t_pos = termites[i].idx;
    if ((t_pos - N) >= 0 && grid[t_pos - N].chip == 1) {
        put_down_chip(i, grid, termites, rank);
        goto end;
    }
    if ((t_pos - 1) % N != (N - 1) && t_pos - 1 >= 0 && grid[t_pos - 1].chip == 1) {
        put_down_chip(i, grid, termites, rank);
        goto end;
    }
    if ((t_pos + N) < (N * N) && grid[t_pos + N].chip == 1) {
        put_down_chip(i, grid, termites, rank);
        goto end;
    }
    if ((t_pos + 1) % N != 0 && grid[t_pos + 1].chip == 1) {
        put_down_chip(i, grid, termites, rank);
        goto end;
    }

    wiggle(i, grid, termites, rank);
    if (termites[i].idx == t_pos) {
        get_away(i, grid, termites, rank);
    }
    find_new_pile(i, grid, termites, rank);
    
end:
    double end_time = MPI_Wtime();
    find_new_pile_time_total += end_time - start_time;
}

void put_down_chip(int i, struct spot grid[], struct termite termites[], int rank) {
    double start_time = MPI_Wtime();

    int old_t_pos = termites[i].idx;
    grid[old_t_pos].chip = 1;
    termites[i].has_chip = 0;
    termites[i].moves = 1;
    get_away(i, grid, termites, rank);

    double end_time = MPI_Wtime();
    put_down_chip_time_total += end_time - start_time;
}

void get_away(int i, struct spot grid[], struct termite termites[], int rank) {
    double start_time = MPI_Wtime();

    int new_pos = rand() % (N * N);
    while (grid[new_pos].has_t == 1 || grid[new_pos].chip == 1) {
        new_pos = rand() % (N * N);
    }
    int old_t_pos = termites[i].idx;
    grid[old_t_pos].has_t = 0;
    grid[new_pos].has_t = 1;
    termites[i].idx = new_pos;

    double end_time = MPI_Wtime();
    get_away_time_total += end_time - start_time;
}

void wiggle(int i, struct spot grid[], struct termite termites[], int rank) {
    double start_time = MPI_Wtime();

    int arr[4] = {0, 1, 2, 3};
    int new_pos;
    int old_pos = termites[i].idx;
    int dir;
    for (int c = 0; c < 4; c++) {
        dir = arr[rand() % 4];
        switch (dir) {
            case 0:
                new_pos = old_pos - N;
                if (new_pos >= 0) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
                    termites[i].idx = new_pos;
                    goto end;
                }
                continue;
            case 1:
                new_pos = old_pos - 1;
                if ((new_pos % N != (N - 1)) && (new_pos >= 0)) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
                    termites[i].idx = new_pos;
                    goto end;
                }
                continue;
            case 2:
                new_pos = old_pos + N;
                if (new_pos < (N * N)) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
                    termites[i].idx = new_pos;
                    goto end;
                }
                continue;
            case 3:
                new_pos = old_pos + 1;
                if (new_pos % N != 0) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
                    termites[i].idx = new_pos;
                    goto end;
                }
                continue;
        }
    }

end:
    double end_time = MPI_Wtime();
    wiggle_time_total += end_time - start_time;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct spot grid[N*N] = {0};
    struct termite termites[B] = {0};

    if (rank == 0) {
        double start_time = MPI_Wtime();
    }

    setup(grid, termites, rank);

    MPI_Bcast(grid, N*N*sizeof(struct spot), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(termites, B*sizeof(struct termite), MPI_BYTE, 0, MPI_COMM_WORLD);

    int termites_per_proc = B / size;

    double start_time = MPI_Wtime();
    for (int i = 0; i < T; i++) {
        go(grid, termites, rank, size);

        MPI_Allgather(MPI_IN_PLACE, termites_per_proc*sizeof(struct termite), MPI_BYTE,
                      termites, termites_per_proc*sizeof(struct termite), MPI_BYTE,
                      MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Print times
        printf("Total Setup time: %f seconds \n", setup_time_total / size);
        printf("Total Go time: %f seconds \n", go_time_total) / size;
        printf("Total Search for chip time: %f seconds \n", search_for_chip_time_total / size);
        printf("Total Find new pile time: %f seconds \n", find_new_pile_time_total / size);
        printf("Total Put down chip time: %f seconds \n", put_down_chip_time_total / size);
        printf("Total Get away time: %f seconds \n", get_away_time_total / size);
        printf("Total Wiggle time: %f seconds \n", wiggle_time_total / size);
	
	double total_time = setup_time_total + go_time_total + search_for_chip_time_total +
                    find_new_pile_time_total + put_down_chip_time_total +
                    get_away_time_total + wiggle_time_total;
        printf("TOTAL : %f seconds \n", total_time / size);	
    }

    MPI_Finalize();
    return 0;
}

