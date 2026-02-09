#ifndef TERMITES_H
#define TERMITES_H
void print_progress(struct spot* grid, int step);
void setup(struct spot grid[], struct termite termites[], int rank);
void go(struct spot grid[], struct termite termites[], int rank, int size);
void update_grid_pos(int s, int new_pos, struct spot grid[], struct termite termites[]);
void search_for_chip(int i, struct spot grid[], struct termite termites[], int rank);
void find_new_pile(int i, struct spot grid[], struct termite termites[], int rank);
void put_down_chip(int i, struct spot grid[], struct termite termites[], int rank);
void get_away(int i, struct spot grid[], struct termite termites[], int rank);
void wiggle(int i, struct spot grid[], struct termite termites[], int rank);
#endif
