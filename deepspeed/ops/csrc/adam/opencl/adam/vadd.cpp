#include "vadd.h"
#include "util.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <iostream>
#include <filesystem>

#define CHECK_ERROR(err)                                                       \
	if (err != CL_SUCCESS) {                                                     \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);              \
		exit(EXIT_FAILURE);                                                        \
	}

#define TILE (128 * 1024 * 1024)

static cl_int err;
static cl_platform_id platforms[16];
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;

static cl_mem d_params, d_grads, d_exp_avg, d_exp_avg_sq;
	
void adam_gpu(
		float* _params,
		const float* grads,
		float* _exp_avg,
		float* _exp_avg_sq,

		size_t _param_size,

		float step_size, float w_decay,

		float _betta1, float _betta2,
		float _eps, 
		float _bias_correction1, float _bias_correction2
		){

	err = clEnqueueWriteBuffer(queue, d_params, CL_TRUE, 0, _param_size * sizeof(float), _params, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, d_grads, CL_TRUE,  0, _param_size * sizeof(float), grads, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, d_exp_avg, CL_TRUE,  0, _param_size * sizeof(float), _exp_avg, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, d_exp_avg_sq, CL_TRUE,  0, _param_size * sizeof(float), _exp_avg_sq, 0, NULL, NULL); CHECK_ERROR(err);
	
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_params); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_grads); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_exp_avg); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_exp_avg_sq); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 4, sizeof(size_t), &_param_size ); CHECK_ERROR(err);
	
	err = clSetKernelArg(kernel, 5, sizeof(float), &step_size); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 6, sizeof(float), &w_decay); CHECK_ERROR(err);

	err = clSetKernelArg(kernel, 7, sizeof(float), &_betta1); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 8, sizeof(float), &_betta2); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 9, sizeof(float), &_eps); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 10, sizeof(float), &_bias_correction1); CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 11, sizeof(float), &_bias_correction2); CHECK_ERROR(err);
		
	size_t localSize = 1024;
	size_t globalSize = _param_size;
	
	cl_event event;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL); CHECK_ERROR(err);
	
	clFlush(queue);
	
	err = clEnqueueReadBuffer(queue, d_params, CL_TRUE,     0, _param_size * sizeof(float), _params, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, d_exp_avg, CL_TRUE,    0, _param_size * sizeof(float), _exp_avg, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, d_exp_avg_sq, CL_TRUE, 0, _param_size * sizeof(float), _exp_avg_sq, 0, NULL, NULL); CHECK_ERROR(err);
}

void adam_cpu(
		float* _params,
		float* grads,
		float* _exp_avg,
		float* _exp_avg_sq,
		size_t _param_size
		){
	
	float _alpha = 1e-3;
	float _betta1 = 0.9;
	float _betta2 = 0.999;
	float _eps = 1e-8;
	float _weight_decay = 0;
	float _bias_correction1 = 1.0f;
	float _bias_correction2 = 1.0f;

	float step_size = -1 * _alpha / _bias_correction1;
	float w_decay = -1 * _alpha * _weight_decay;
	
#pragma omp parallel for
	for (size_t k = 0; k < _param_size; ++k ) {
            float grad = grads[k];
			float param =_params[k];
			float momentum = _exp_avg[k];
			float variance = _exp_avg_sq[k];

			momentum = momentum * _betta1;
			momentum = grad * (1 - _betta1) + momentum;

			variance = variance * _betta2;
			grad = grad * grad;
			variance = grad * (1 - _betta2) + variance;

			grad = sqrt(variance);
			grad = grad * _bias_correction2 + _eps;
			grad = momentum / grad;
			
			param += w_decay * param; //AdamW
			param = grad * step_size + param;
			
			_params[k] = param;
			_exp_avg[k] = momentum;
			_exp_avg_sq[k] = variance;
	}
}

void check_adam(
		float* params,
		float* grads,
		float* exp_avg,
		float* exp_avg_sq,

		float* _params,
		float* _exp_avg,
		float* _exp_avg_sq,
		size_t _param_size
		){

	printf("Validating...\n");
	
	float _alpha = 1e-3;
	float _betta1 = 0.9;
	float _betta2 = 0.999;
	float _eps = 1e-8;
	float _weight_decay = 0;
	float _bias_correction1 = 1.0f;
	float _bias_correction2 = 1.0f;
	float step_size = -1 * _alpha / _bias_correction1;
	float w_decay = -1 * _alpha * _weight_decay;

#pragma omp parallel for
	for (size_t k = 0; k < _param_size; ++k ) {
            float grad = grads[k];
			float param = params[k];
			float momentum = exp_avg[k];
			float variance = exp_avg_sq[k];

			momentum = momentum * _betta1;
			momentum = grad * (1 - _betta1) + momentum;

			variance = variance * _betta2;
			grad = grad * grad;
			variance = grad * (1 - _betta2) + variance;

			grad = sqrt(variance);
			grad = grad * _bias_correction2 + _eps;
			grad = momentum / grad;
			param += w_decay * param; //AdamW
			param = grad * step_size + param;

			params[k] = param;
			exp_avg[k] = momentum;
			exp_avg_sq[k] = variance;
	}

	bool is_valid = true;
	int cnt = 0, thr = 10;
	float eps = 1e-2;
	
	for (size_t i = 0; i < _param_size; ++i) {
		float c = _params[i ];
		float c_ans = params[i ];
		if (fabsf(c - c_ans) > eps &&
				(c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
			++cnt;
			if (cnt <= thr)
				printf("params[%ld]: correct_value = %f, your_value = %f\n", i, 
						c_ans, c);
			if (cnt == thr + 1)
				printf("Too many error, only first %d values are printed.\n", thr);
			is_valid = false;
		}
	}
	
	for (size_t i = 0; i < _param_size; ++i) {
		float c = _exp_avg[i ];
		float c_ans = exp_avg[i ];
		if (fabsf(c - c_ans) > eps &&
				(c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
			++cnt;
			if (cnt <= thr)
				printf("exp_avg[%ld]: correct_value = %f, your_value = %f\n", i, 
						c_ans, c);
			if (cnt == thr + 1)
				printf("Too many error, only first %d values are printed.\n", thr);
			is_valid = false;
		}
	}
	
	for (size_t i = 0; i < _param_size; ++i) {
		float c = _exp_avg_sq[i ];
		float c_ans = exp_avg_sq[i ];
		if (fabsf(c - c_ans) > eps &&
				(c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
			++cnt;
			if (cnt <= thr)
				printf("exp_avg_sq[%ld]: correct_value = %.6f, your_value = %.6f\n", i, 
						c_ans, c);
			if (cnt == thr + 1)
				printf("Too many error, only first %d values are printed.\n", thr);
			is_valid = false;
		}
	}

	if (is_valid) {
		printf("Result: VALID\n");
	} else {
		printf("Result: INVALID\n");
	}
}





static void print_platform_info(cl_platform_id platforms[16], cl_uint num_platforms) {
	size_t sz;
	char *buf;
	std::cout << "Total # of platforms : " << num_platforms << std::endl;
	for (size_t i = 0 ; i< num_platforms; i++){
		cl_platform_id platform = platforms[i];
		cl_uint numDevices;
		CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
		buf = (char *)malloc(sz);
		CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
		
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
	
		printf("Detected OpenCL platform [ %s ] has %d devices.\n", buf, numDevices);

		free(buf);
	}
}

static void print_device_info(cl_device_id device) {
	size_t sz;
	char *buf;
	CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
	buf = (char *)malloc(sz);
	CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
	printf("Detected OpenCL device: %s\n", buf);
	free(buf);
}

static cl_program create_and_build_program_with_source(cl_context context,
		cl_device_id device,
		const char *file_name) {
	FILE *file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("Failed to open %s\n", file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	size_t source_size = ftell(file);
	rewind(file);
	char *source_code = (char *)malloc(source_size + 1);
	size_t ntotal = 0;
	while (ntotal < source_size) {
		int nread = fread(source_code, sizeof(char), source_size, file);
		ntotal += nread;
	}
	source_code[source_size] = '\0';
	fclose(file);
	cl_program program = clCreateProgramWithSource(
			context, 1, (const char **)&source_code, &source_size, &err);
	CHECK_ERROR(err);
	free(source_code);
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
					NULL, &log_size));
		char *log = (char *)malloc(log_size + 1);
		CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
					log_size, log, NULL));
		log[log_size] = 0;
		printf("Compile error:\n%s\n", log);
		free(log);
	}
	CHECK_ERROR(err);
	return program;
}

void adam_initialize() {
	// Get OpenCL platform
	cl_uint num_platforms;
	err = clGetPlatformIDs(16, platforms, &num_platforms);
	CHECK_ERROR(err);
	print_platform_info(platforms, num_platforms);

	cl_platform_id platform;

	for( cl_uint i =0; i< num_platforms; ++i){
		cl_uint numDevices;
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
		if ( numDevices > 0){
			platform = platforms[i];
			break;
		}
	}
		
	// Get OpenCL device (only 1)
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	
	CHECK_ERROR(err);
	print_device_info(device);

	// Create OpenCL context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	// Create OpenCL command queue
	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);
	
	// std::cout << "Current working dir:" << std::filesystem::current_path() <<std::endl;
	// Compile program from "kernel.cl"
	program = create_and_build_program_with_source(context, device, "kernel.cl");

	// Extract kernel from compiled program
	kernel = clCreateKernel(program, "vadd", &err);
	CHECK_ERROR(err);
}

void adam_create_buf(size_t _param_size){
	// Create GPU buffers
	
	d_params = clCreateBuffer(context, CL_MEM_READ_WRITE, _param_size * sizeof(float), NULL, &err);
	CHECK_ERROR(err);
	d_grads = clCreateBuffer(context, CL_MEM_READ_ONLY, _param_size * sizeof(float), NULL, &err);
	CHECK_ERROR(err);
	d_exp_avg = clCreateBuffer(context, CL_MEM_READ_WRITE, _param_size * sizeof(float), NULL, &err);
	CHECK_ERROR(err);
	d_exp_avg_sq = clCreateBuffer(context, CL_MEM_READ_WRITE, _param_size * sizeof(float), NULL, &err);
	CHECK_ERROR(err);
}

void adam_finalize() {
	clReleaseMemObject(d_params);
	clReleaseMemObject(d_grads);
	clReleaseMemObject(d_exp_avg);
	clReleaseMemObject(d_exp_avg_sq);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
