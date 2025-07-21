#include "util.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

static double start_time[8];

void timer_init() { srand(time(NULL)); }

static double get_time() {
	struct timespec tv;
	clock_gettime(CLOCK_MONOTONIC, &tv);
	return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

void alloc_mat(float **m, int R, int S) {
	*m = (float *)aligned_alloc(32, sizeof(float) * R * S);
	if (*m == NULL) {
		printf("Failed to allocate memory for mat.\n");
		exit(0);
	}
}

void rand_mat(float *m, int R, int S) {
	int N = R * S;
	for (int j = 0; j < N; j++) {
		m[j] = (float)rand() / RAND_MAX - 0.5;
	}
}

void zero_mat(float *m, int R, int S) {
	int N = R * S;
	memset(m, 0, sizeof(float) * N);
}

void print_mat(float *m, int M, int N) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%+.5f ", m[i * N + j]);
		}
		printf("\n");
	}
}


