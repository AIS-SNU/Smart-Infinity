#pragma once

void timer_init();

void timer_start(int i);

double timer_stop(int i);

void alloc_mat(float **m, int R, int S);

void rand_mat(float *m, int R, int S);

void zero_mat(float *m, int R, int S);

void print_vec(float *m, int N);

void print_mat(float *m, int M, int N);

