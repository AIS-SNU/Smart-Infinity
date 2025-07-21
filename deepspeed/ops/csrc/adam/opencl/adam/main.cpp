#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include "vadd.h"
#include "util.h"

#define TILE (128 * 1024 * 1024)

static void print_help(const char *prog_name) {
	printf("Usage: %s [-pvh] [-n num_iterations] M N K\n", prog_name);
	printf("Options:\n");
	printf("     -p : print vector. (default: off)\n");
	printf("     -v : validate vector dot. (default: off)\n");
	printf("     -h : print this page.\n");
	printf("     -n : number of iterations (default: 1)\n");
	printf("      M : number of rows of matrix A and C. (default: 8)\n");
	printf("      N : number of columns of matrix B and C. (default: 8)\n");
	printf(
			"      K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_data = false;
static bool validation = false;
static unsigned int M = 1048576;

static int num_iterations = 3;

static void parse_opt(int argc, char **argv) {
	int c;
	while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
		switch (c) {
			case 'p':
				print_data = true;
				break;
			case 'v':
				validation = true;
				break;
			case 'n':
				num_iterations = atoi(optarg);
				break;
			case 'h':
			default:
				print_help(argv[0]);
				exit(0);
		}
	}
	for (int i = optind, j = 0; i < argc; ++i, ++j) {
		switch (j) {
			case 0:
				M = atoi(argv[i]);
				break;
			case 1:
				//N = atoi(argv[i]);
				break;
			case 2:
				//K = atoi(argv[i]);
				break;
			default:
				break;
		}
	}
	printf("Options:\n");
	//printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
	printf("  Problem size: M = %d \n", M);
	printf("  Number of iterations: %d\n", num_iterations);
	printf("  Print matrix: %s\n", print_data ? "on" : "off");
	printf("  Validation: %s\n", validation ? "on" : "off");
	printf("\n");
}

int main(int argc, char **argv) {

	parse_opt(argc, argv);

	printf("Initializing... "); fflush(stdout);
	float *params, *grads, *exp_avg, *exp_avg_sq;
	float *_params, *_exp_avg, *_exp_avg_sq;

	// Initialize random seed
	timer_init();

	// Allocate vectors
	alloc_mat(&params     , M, 1);
	alloc_mat(&grads      , M, 1);
	alloc_mat(&exp_avg    , M, 1);
	alloc_mat(&exp_avg_sq , M, 1);
	
	alloc_mat(&_params    , M, 1);
	alloc_mat(&_exp_avg   , M, 1);
	alloc_mat(&_exp_avg_sq, M, 1);

	// Set each element to a random value
	rand_mat(params, M, 1);
	rand_mat(grads, M, 1);
	zero_mat(exp_avg, M, 1);
	zero_mat(exp_avg_sq, M, 1);

	memcpy(_params, params, sizeof(float) *M);
	memcpy(_exp_avg, exp_avg,sizeof(float) * M);
	memcpy(_exp_avg_sq, exp_avg_sq,sizeof(float) * M);

	printf("done!\n");

	// Initialize OpenCL
	printf("Initializing OpenCL...\n");
	fflush(stdout);

	adam_initialize();
	adam_create_buf(TILE);

	float _alpha = 1e-3;
    float _weight_decay = 0;

	float _bias_correction1 = 1.0f;
	float step_size = -1 * _alpha / _bias_correction1;
	float w_decay = -1 * _alpha * _weight_decay;

	// Few warmup iterations
	for (int i = 0; i < 3; i++) {
		//adam_cpu(_params, grads, _exp_avg, _exp_avg_sq, M);
		//adam_gpu(_params, grads, _exp_avg, _exp_avg_sq, M);
	}

	double elapsed_time_sum = 0;
	for (int i = 0; i < num_iterations; ++i) {
		printf("Calculating...(iter=%d) ", i);
		fflush(stdout);
		memcpy(_params, params, sizeof(float) * M);
		memcpy(_exp_avg, exp_avg, sizeof(float) * M);
		memcpy(_exp_avg_sq, exp_avg_sq, sizeof(float) * M);

		timer_start(0);
		
		for (unsigned int idx = 0; idx < M  ; idx+= TILE){
			adam_gpu(_params + idx, grads + idx, _exp_avg + idx, _exp_avg_sq + idx, TILE, step_size, w_decay);
			//adam_cpu(_params + idx, grads + idx, _exp_avg + idx, _exp_avg_sq + idx, BUF_SIZE);
		}

		double elapsed_time = timer_stop(0);
		printf("%f sec\n", elapsed_time);
		elapsed_time_sum += elapsed_time;
	}

	if (print_data) {
		printf("MATRIX param:\n");
		print_mat(params, M, 1);
		printf("MATRIX grads:\n");
		print_mat(grads, M, 1);
		printf("MATRIX exp_avg:\n");
		print_mat(exp_avg, M, 1);
		printf("MATRIX exp_avg_sq:\n");
		print_mat(exp_avg_sq, M, 1);

		printf("MATRIX _params:\n");
		print_mat(_params, M, 1);
		printf("MATRIX _exp_avg:\n");
		print_mat(_exp_avg, M, 1);
		printf("MATRIX _exp_avg_sq:\n");
		print_mat(_exp_avg_sq, M, 1);
	}

	// Finalize OpenCL
	adam_finalize();

	if (validation) {
		check_adam(params, grads, exp_avg, exp_avg_sq, _params, _exp_avg, _exp_avg_sq, M);
	}

	double elapsed_time_avg = elapsed_time_sum / num_iterations;
	printf("Avg. time: %f sec\n", elapsed_time_avg);
	printf("Avg. throughput: %f GFLOPS\n",
			2.0 * M / elapsed_time_avg / 1e9);

	return 0;
}
