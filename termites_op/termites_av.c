
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "termites.h"

#define B	20	// num of termites
#define T	10000	// num of steps
#define N	20	// nxn grid size
#define D	20	// density (%) of chips compared to grid

struct termite {
    int    has_chip;
    int    moves;
    int    idx;
};

struct spot {
    int    chip;
    int    has_t;
};

int main() {
    struct spot grid[N*N] = {0};
    struct termite termites[B] = {0};
   
    setup(grid, termites);
    int i = 0;
    print_progress(grid, 0);
    while (i < T) {
	go(grid, termites);
	i++;
    }
    print_progress(grid, T);
   
    return 0;
}

void setup(struct spot grid[], struct termite termites[]) {
    //Init rand() seed
    srand(time(NULL));
    
    //Init grid values
    int grid_size = N*N;
    for (int i=0;i<grid_size;i++) {
        grid[i].chip = 0;
	grid[i].has_t = 0;
    }    

    //Randomly place chips
    int chip_amount = (grid_size) * (D*0.01);
    int i = rand() % (grid_size);
    while (chip_amount > 0) {
        i = rand() % (grid_size);
        if (grid[i].chip != 1 && chip_amount != 0) {
	    grid[i].chip = 1;
	    chip_amount = chip_amount-1;     
	}
    }
    
    // Randomly place termites
    for (int b=0;b<B;b++) {   

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

void print_progress(struct spot grid[], int step) {
	printf("Step: %d\n", step);
	for (int j=0;j<N;j++){	    
	    for (int k=0;k<N;k++) {
		if (grid[k + (j*N)].has_t == 1){
		        printf(" T ");
		} else if (grid[k + (j*N)].chip) {
			printf(" C ");
		}
		else {
		    printf(" . ");
		}
            }
	printf("\n");
	}
}


void go(struct spot grid[], struct termite termites[]) {
    // Termite movement
    for (int s = 0; s < B; s++) {
        // Skip termites that have already moved
        if (termites[s].moves == 1) {
            continue;
        }
        // If a termite has a chip find a place to drop it
        // Otherwise search for a chip
        else if (termites[s].has_chip == 1) {
	    find_new_pile(s, grid, termites);
        } else {
            search_for_chip(s, grid, termites);
        }
    }
    for (int m=0; m<B;m++) {
        termites[m].moves = 0;
    }
}

void update_grid_pos(int s, int new_pos, 
	struct spot grid[], struct termite termites[]) {
    // Remove old grid pos
    int old_pos = termites[s].idx;
    grid[old_pos].has_t = 0;
    
    // Update to new grid pos
    termites[s].has_chip = 1;
    termites[s].moves = 1;
    termites[s].idx = new_pos;

    grid[new_pos].chip = 0;
    grid[new_pos].has_t = 1;
}

void search_for_chip(int i, struct spot grid[], struct termite termites[]) {
    int t_pos = termites[i].idx;
    // Check up
    if (( (t_pos - N) >= 0) && (grid[t_pos-N].chip == 1)) {
        update_grid_pos(i, (t_pos - N), grid, termites);
	return; 
    } 
    // Check left
    else if (( (t_pos - 1) % N != (N-1)) && (t_pos - 1 > 0 ) && (grid[t_pos-1].chip == 1)) {
        update_grid_pos(i, (t_pos - 1), grid, termites);
	return;
    }  
    // Check down
    else if (( (t_pos + N) < (N*N)) && (grid[t_pos+N].chip == 1)) {
        update_grid_pos(i, (t_pos + N), grid, termites);
	return;
    } 
    // Check right 
    else if ( ((t_pos + 1) % N != 0) && (grid[t_pos+1].chip == 1)) {
        update_grid_pos(i, (t_pos + 1), grid, termites);
	return;
    } else {
        // No Chips????? go random pos on grid and try again
        wiggle(i, grid, termites);
        // Check if termite stuck 
        if (termites[i].idx == t_pos) {
            get_away(i, grid, termites);
	}
        search_for_chip(i, grid, termites);
    } 
}

void find_new_pile(int i, struct spot grid[], struct termite termites[]) { 
    int t_pos = termites[i].idx;
    // Check up
    if (( (t_pos - N) >= 0) && (grid[t_pos-N].chip == 1)) {
        put_down_chip(i, grid, termites); 
    } 
    // Check left
    else if (( (t_pos - 1) % N != (N-1)) && (t_pos - 1 > 0 ) && (grid[t_pos-1].chip == 1)) {
        put_down_chip(i, grid, termites); 
    }  
    // Check down
    else if (( (t_pos + N) < (N*N)) && (grid[t_pos+N].chip == 1)) {
        put_down_chip(i, grid, termites); 
    } 
    // Check right 
    else if ( ((t_pos + 1) % N != 0) && (grid[t_pos+1].chip == 1)) {
        put_down_chip(i, grid, termites); 
    } else {
        wiggle(i, grid, termites);
        // Check if termite stuck
        if (termites[i].idx == t_pos) {
            get_away(i, grid, termites);
	}
	find_new_pile(i, grid, termites);    
    }
    
}

void put_down_chip(int i, struct spot grid[], struct termite termites[]){
    // Put down chip
    int old_t_pos = termites[i].idx;
    grid[old_t_pos].chip = 1;
    // Update termite stats
    termites[i].has_chip = 0;
    // Get_away
    termites[i].moves = 1;
    get_away(i, grid, termites);
    
    return;
}

void get_away(int i, struct spot grid[], struct termite termites[]) {
    //Place termite in random position in grid
    int new_pos = rand() % (N*N);
    
    while((grid[new_pos].has_t == 1) || (grid[new_pos].chip == 1)) {
        new_pos = rand() % (N*N);
    }
    int old_t_pos = termites[i].idx;
    
    grid[old_t_pos].has_t = 0;
    grid[new_pos].has_t = 1;
    termites[i].idx = new_pos;
    return;
}


void wiggle(int i, struct spot grid[], struct termite termites[]) {
    int arr[4] = {1,2,3,4};
    int new_pos;
    int old_pos = termites[i].idx;
    int dir;
    for (int c=0; c<4;c++) {
        dir = arr[rand() % 4];
        switch (dir) {
            case 0:
                new_pos = old_pos-N;
                if ( (new_pos) >= 0) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
    		    termites[i].idx = new_pos;
                    return;
                }
                continue;
            case 1:
                new_pos = old_pos-1;
                if (( (new_pos) % N != (N-1)) && (new_pos > 0 )){
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
	  	    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
    		    termites[i].idx = new_pos;
                    return;
               }
               continue;
            case 2:
                new_pos = old_pos+N;
                if ( (new_pos) < (N*N)) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
    		    termites[i].idx = new_pos;
                    return;
                }
                continue;
            case 3:
	        new_pos = i+1;
                if ( (new_pos) % N != 0) {
                    if ((grid[new_pos].chip == 1) || grid[new_pos].has_t == 1) {
                        continue;
                    }
                    grid[old_pos].has_t = 0;
                    grid[new_pos].has_t = 1;
    		    termites[i].idx = new_pos;
                    return;
                }
                continue;
        }
    }
}



