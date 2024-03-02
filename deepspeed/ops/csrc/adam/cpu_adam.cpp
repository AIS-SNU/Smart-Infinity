// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"
#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <vector>
#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <CL/cl_ext_xilinx.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <cmath>
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <cassert>
#include <CL/cl_ext.h>

#include <thread>

# define DS_CPU 0
# define CPU 1
# define GPU 2
# define FPGA 3
# define ACC_TYPE DS_CPU
#include "vadd.h"

#define TILE (128 * 1024 * 1024)

template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

#include <x86intrin.h>

#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }                                       

#define MAX_DEVICE 16

std::vector<cl::Platform> platforms[MAX_DEVICE];
std::vector<cl::Device> devices[MAX_DEVICE];
static bool init[MAX_DEVICE] = {false,};
char *file_bufs[MAX_DEVICE];
cl::Program::Binaries bins[MAX_DEVICE];
cl::Context contexts[MAX_DEVICE];
cl::Program programs[MAX_DEVICE];
cl::CommandQueue queues[MAX_DEVICE];
cl::Kernel krnls[MAX_DEVICE];

cl::Buffer param16_pool[MAX_DEVICE];
cl::Buffer param_pool[MAX_DEVICE];
cl::Buffer exp_avg_pool[MAX_DEVICE];
cl::Buffer exp_avg_sq_pool[MAX_DEVICE];
cl::Buffer grad_pool[MAX_DEVICE];
cl::Buffer grad_idx_pool[MAX_DEVICE];
cl::Buffer grad_val_pool[MAX_DEVICE];

float* _p2p_params[MAX_DEVICE] = {nullptr, };
float* p2p_params[MAX_DEVICE] = {nullptr, };
float* p2p_exp_avgs[MAX_DEVICE] = {nullptr, };
float* p2p_exp_avg_sqs[MAX_DEVICE] = {nullptr, };
half* p2p_grads[MAX_DEVICE] = {nullptr, };
int* p2p_grads_idx[MAX_DEVICE] = {nullptr, };
half* p2p_grads_val[MAX_DEVICE] = {nullptr, };

int nvmeFd_exp_avgs[MAX_DEVICE] = {-1, };
int nvmeFd_exp_avg_sqs[MAX_DEVICE] = { -1, };

std::vector<std::thread> threads;

std::thread* write_param[MAX_DEVICE] = { nullptr,  };
std::thread* write_exp_avg[MAX_DEVICE] = { nullptr,  };
std::thread* write_exp_avg_sq[MAX_DEVICE] = {nullptr, };

void sync_thread()
{
	for ( auto& thread : threads)
	{
		thread.join();
	}
	threads.clear();
}

void write_thread(int nvmeFd, float* p2p_ptr, size_t nbytes, bool cls)
{
	int ret = - 1;
	ret = pwrite(nvmeFd, (void*)p2p_ptr, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: write() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	if (cls) { (void)close(nvmeFd); }
}

void thread_work_adagrad_comp(
		std::string param_path,
		std::string exp_avg_sq_path,
		std::string grad_path,
		size_t _param_size ,
		float combined_unscale,
		half* fp16_params_ptr,
		int device_id,
		float _alpha,
	    float _eps,
	    float _weight_decay,
		float compression_ratio
	)
{
	assert( int(_param_size) % (16) == 0);
	
	auto context = contexts[device_id];
	auto q = queues[device_id];

	auto krnl_adam = krnls[device_id];	

	int ret;

	size_t nbytes = _param_size * sizeof(float);
	
	float step_size = -1 * _alpha;
	float w_decay = _weight_decay;

	cl_int err;
			
	int comp_grad_size = int (_param_size * compression_ratio);
	int padded_comp_grad_size = ((comp_grad_size - 1)/1024 + 1 ) * 1024;
	int comp_nbytes = padded_comp_grad_size * sizeof(float);

	int cnt = 2;
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, int(padded_comp_grad_size)));
	cnt++; // grad16
	cnt++; // param16
	cnt++; // param
	cnt++; // exp_avg_sq
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, int(_param_size)));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _eps));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, w_decay));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, step_size));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, combined_unscale));

	size_t offset = 0;	
	
	int* p2p_grad_idx = p2p_grads_idx[device_id];	
	half* p2p_grad_val = p2p_grads_val[device_id];	
	float* p2p_param = p2p_params[device_id];	
	float* p2p_exp_avg_sq = p2p_exp_avg_sqs[device_id];	

	int nvmeFd_grad_idx = -1;
	int nvmeFd_grad_val = -1;
	int nvmeFd_param = -1;
	int nvmeFd_exp_avg_sq = nvmeFd_exp_avg_sqs[device_id];

	std::string idx_offset_str = "0";
	std::string val_offset_str = "1";

	nvmeFd_grad_idx = open((grad_path + idx_offset_str).c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_grad_val = open((grad_path + val_offset_str).c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);

	nvmeFd_param = open(param_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_exp_avg_sq = open(exp_avg_sq_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);


	if (nvmeFd_grad_val < 0 || nvmeFd_grad_idx < 0){
		std::cerr << "ERROR: open " << grad_path << " failed with " << std::endl;
		assert(false);
	}
	
	//q.enqueueWriteBuffer ( grad_idx_pool[device_id], CL_FALSE, 0, comp_nbytes, grad_idx_ptr);

		
	ret = pread(nvmeFd_grad_idx, (void*)p2p_grad_idx, comp_nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
	ret = pread(nvmeFd_grad_val, (void*)p2p_grad_val, comp_nbytes/2, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
	if (write_param[device_id] != nullptr) {  write_param[device_id]->join(); write_param[device_id] = nullptr; }
	ret = pread(nvmeFd_param, (void*)p2p_param, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg_sq[device_id] != nullptr) {  write_exp_avg_sq[device_id]->join(); write_exp_avg_sq[device_id] = nullptr ;}
	ret = pread(nvmeFd_exp_avg_sq, (void*)p2p_exp_avg_sq, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

    //Launch the Kernel
	q.enqueueTask(krnl_adam, nullptr, nullptr);
	
	q.enqueueReadBuffer ( param16_pool[device_id], CL_FALSE, 0, nbytes/2, fp16_params_ptr);
	q.finish();

	(void)close(nvmeFd_grad_idx);
	(void)close(nvmeFd_grad_val);
	
	write_param[device_id] = new std::thread(write_thread, nvmeFd_param, p2p_param, nbytes, true);
	write_exp_avg_sq[device_id] = new std::thread(write_thread, nvmeFd_exp_avg_sq, p2p_exp_avg_sq, nbytes, true );

}
void thread_work_sgd_comp(
		std::string param_path,
		std::string exp_avg_path,
		std::string grad_path,
		size_t _param_size ,
		float combined_unscale,
		half* fp16_params_ptr,
		int device_id,
		float _alpha,
	    float _betta1,
	    float _weight_decay,
		float compression_ratio,
		int* grad_idx_ptr
	)
{
	assert( int(_param_size) % (16) == 0);
	
	auto context = contexts[device_id];
	auto q = queues[device_id];

	auto krnl_adam = krnls[device_id];	

	int ret;

	size_t nbytes = _param_size * sizeof(float);
	
	float step_size = -1 * _alpha ;
	float w_decay = _weight_decay;

	cl_int err;
			
	int comp_grad_size = int (_param_size * compression_ratio);
	int padded_comp_grad_size = ((comp_grad_size - 1)/1024 + 1 ) * 1024;
	int comp_nbytes = padded_comp_grad_size * sizeof(float);

	int cnt = 2;
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, int(padded_comp_grad_size)));
	cnt++; // grad16
	cnt++; // param16
	cnt++; // param
	cnt++; // exp_avg
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, int(_param_size)));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _betta1));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, w_decay));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, step_size));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, combined_unscale));

	size_t offset = 0;	
	
	int* p2p_grad_idx = p2p_grads_idx[device_id];	
	half* p2p_grad_val = p2p_grads_val[device_id];	
	float* p2p_param = p2p_params[device_id];	
	float* p2p_exp_avg = p2p_exp_avgs[device_id];	

	int nvmeFd_grad_idx = -1;
	int nvmeFd_grad_val = -1;
	int nvmeFd_param = -1;
	int nvmeFd_exp_avg = nvmeFd_exp_avgs[device_id];

	std::string idx_offset_str = "0";
	std::string val_offset_str = "1";

	nvmeFd_grad_idx = open((grad_path + idx_offset_str).c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_grad_val = open((grad_path + val_offset_str).c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);

	nvmeFd_param = open(param_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_exp_avg = open(exp_avg_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);


	if (nvmeFd_grad_val < 0 || nvmeFd_grad_idx < 0){
		std::cerr << "ERROR: open " << grad_path << " failed with " << std::endl;
		assert(false);
	}
	
		
	ret = pread(nvmeFd_grad_idx, (void*)p2p_grad_idx, comp_nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
	ret = pread(nvmeFd_grad_val, (void*)p2p_grad_val, comp_nbytes/2, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
	if (write_param[device_id] != nullptr) {  write_param[device_id]->join(); write_param[device_id] = nullptr; }
	ret = pread(nvmeFd_param, (void*)p2p_param, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg[device_id] != nullptr) {  write_exp_avg[device_id]->join(); write_exp_avg[device_id] = nullptr; }
	ret = pread(nvmeFd_exp_avg, (void*)p2p_exp_avg, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

    //Launch the Kernel
	q.enqueueTask(krnl_adam, nullptr, nullptr);
	
	q.enqueueReadBuffer ( param16_pool[device_id], CL_FALSE, 0, nbytes/2, fp16_params_ptr);
	q.finish();

	(void)close(nvmeFd_grad_idx);
	(void)close(nvmeFd_grad_val);
	
	write_param[device_id] = new std::thread(write_thread, nvmeFd_param, p2p_param, nbytes, true);
	write_exp_avg[device_id] = new std::thread(write_thread, nvmeFd_exp_avg, p2p_exp_avg, nbytes, true);

}



void thread_work_comp(
		std::string param_path,
		std::string exp_avg_path,
		std::string exp_avg_sq_path,
		std::string grad_path,
		size_t _param_size ,
		float combined_unscale,
		half* fp16_params_ptr,
		int device_id,
		float _alpha,
	    float _betta1,
	    float _betta2,
	    float _eps,
	    float _weight_decay,
		float _bias_correction1,
        float _bias_correction2,
		float compression_ratio,
		int* grad_idx_ptr
	)
{
	assert( int(_param_size) % (16) == 0);
	
	auto context = contexts[device_id];
	auto q = queues[device_id];

	auto krnl_adam = krnls[device_id];	

	int ret;

	size_t nbytes = _param_size * sizeof(float);
	
	float step_size = -1 * _alpha / _bias_correction1;
	float w_decay = -1 * _alpha * _weight_decay;

	cl_int err;
			
	int comp_grad_size = int (_param_size * compression_ratio);
	int padded_comp_grad_size = ((comp_grad_size - 1)/1024 + 1 ) * 1024;
	int comp_nbytes = padded_comp_grad_size * sizeof(float);

	int cnt = 2;
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, int(padded_comp_grad_size)));
	cnt++; // grad16
	cnt++; // param16
	cnt++; // param
	cnt++; // exp_avg
	cnt++; // exp_avg_sq
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, int(_param_size)));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _betta1));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _betta2));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _bias_correction2));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _eps));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, w_decay));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, step_size));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, combined_unscale));

	size_t offset = 0;	
	
	int* p2p_grad_idx = p2p_grads_idx[device_id];	
	half* p2p_grad_val = p2p_grads_val[device_id];	
	float* p2p_param = p2p_params[device_id];	
	float* p2p_exp_avg = p2p_exp_avgs[device_id];	
	float* p2p_exp_avg_sq = p2p_exp_avg_sqs[device_id];	

	int nvmeFd_grad_idx = -1;
	int nvmeFd_grad_val = -1;
	int nvmeFd_param = -1;
	int nvmeFd_exp_avg = nvmeFd_exp_avgs[device_id];
	int nvmeFd_exp_avg_sq = nvmeFd_exp_avg_sqs[device_id];

	std::string idx_offset_str = "0";
	std::string val_offset_str = "1";

	nvmeFd_grad_idx = open((grad_path + idx_offset_str).c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_grad_val = open((grad_path + val_offset_str).c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);

	nvmeFd_param = open(param_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_exp_avg = open(exp_avg_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);
	nvmeFd_exp_avg_sq = open(exp_avg_sq_path.c_str(), O_RDWR | O_SYNC | O_CREAT | O_DIRECT, 0644);


	if (nvmeFd_grad_val < 0 || nvmeFd_grad_idx < 0){
		std::cerr << "ERROR: open " << grad_path << " failed with " << std::endl;
		assert(false);
	}
	
		
	ret = pread(nvmeFd_grad_idx, (void*)p2p_grad_idx, comp_nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
	ret = pread(nvmeFd_grad_val, (void*)p2p_grad_val, comp_nbytes/2, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
	if (write_param[device_id] != nullptr) {  write_param[device_id]->join(); write_param[device_id] = nullptr; }
	ret = pread(nvmeFd_param, (void*)p2p_param, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg[device_id] != nullptr) {  write_exp_avg[device_id]->join(); write_exp_avg[device_id] = nullptr; }
	ret = pread(nvmeFd_exp_avg, (void*)p2p_exp_avg, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg_sq[device_id] != nullptr) {  write_exp_avg_sq[device_id]->join(); write_exp_avg_sq[device_id] = nullptr ;}
	ret = pread(nvmeFd_exp_avg_sq, (void*)p2p_exp_avg_sq, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

    //Launch the Kernel
	q.enqueueTask(krnl_adam, nullptr, nullptr);
	
	q.enqueueReadBuffer ( param16_pool[device_id], CL_FALSE, 0, nbytes/2, fp16_params_ptr);
	q.finish();

	(void)close(nvmeFd_grad_idx);
	(void)close(nvmeFd_grad_val);
	
	write_param[device_id] = new std::thread(write_thread, nvmeFd_param, p2p_param, nbytes, true);
	write_exp_avg[device_id] = new std::thread(write_thread, nvmeFd_exp_avg, p2p_exp_avg, nbytes, true);
	write_exp_avg_sq[device_id] = new std::thread(write_thread, nvmeFd_exp_avg_sq, p2p_exp_avg_sq, nbytes, true );

}

void thread_work_adagrad(
		std::string param_path,
		std::string exp_avg_sq_path,
		std::string grad_path,
		size_t _param_size ,
		float combined_unscale,
		half* fp16_params_ptr,
		int device_id,
		float _alpha,
	    float _eps,
	    float _weight_decay
	)
{
	assert( int(_param_size) % (16) == 0);
	
	auto context = contexts[device_id];
	auto q = queues[device_id];

	auto krnl_adam = krnls[device_id];	

	int ret;

	size_t nbytes = _param_size * sizeof(float);
	
	cl_int err;

	float step_size = -1 * _alpha;

	unsigned int cnt = 4;
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, uint32_t(_param_size)));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _eps));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _weight_decay));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, step_size));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, combined_unscale));

	size_t offset = 0;	

	float* p2p_param = p2p_params[device_id];	
	half* p2p_grad = p2p_grads[device_id];	
	float* p2p_exp_avg_sq = p2p_exp_avg_sqs[device_id];	

	int nvmeFd_param = -1;
	int nvmeFd_exp_avg_sq = -1;
	int nvmeFd_grad = -1;
	

	nvmeFd_grad = open(grad_path.c_str(), O_RDWR | O_SYNC | O_DIRECT, 0644);
	ret = pread(nvmeFd_grad, (void*)p2p_grad, nbytes/2, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_param[device_id] != nullptr) {  write_param[device_id]->join(); write_param[device_id] = nullptr; }
	nvmeFd_param = open(param_path.c_str(), O_RDWR | O_SYNC  | O_DIRECT, 0644);
	ret = pread(nvmeFd_param, (void*)p2p_param, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg_sq[device_id] != nullptr) {  write_exp_avg_sq[device_id]->join(); write_exp_avg_sq[device_id] = nullptr ;}
	nvmeFd_exp_avg_sq = open(exp_avg_sq_path.c_str(), O_RDWR | O_SYNC | O_DIRECT, 0644);
	ret = pread(nvmeFd_exp_avg_sq, (void*)p2p_exp_avg_sq, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
    //Launch the Kernel
	q.enqueueTask(krnl_adam, nullptr, nullptr);
	q.enqueueReadBuffer ( param16_pool[device_id], CL_FALSE, 0, nbytes/2, fp16_params_ptr);
	q.finish();

	(void)close(nvmeFd_grad);
	
	write_param[device_id] = new std::thread(write_thread, nvmeFd_param, p2p_param, nbytes, true);
	
	write_exp_avg_sq[device_id] = new std::thread(write_thread, nvmeFd_exp_avg_sq, p2p_exp_avg_sq, nbytes, true);
	
}

void thread_work_sgd(
		std::string param_path,
		std::string exp_avg_path,
		std::string grad_path,
		size_t _param_size ,
		float combined_unscale,
		half* fp16_params_ptr,
		int device_id,
		float _alpha,
	    float _betta1,
	    float _weight_decay
	)
{
	assert( int(_param_size) % (16) == 0);
	
	auto context = contexts[device_id];
	auto q = queues[device_id];

	auto krnl_adam = krnls[device_id];	

	int ret;

	size_t nbytes = _param_size * sizeof(float);
	
	float step_size = -1 * _alpha;
	float w_decay =  _weight_decay;

	cl_int err;
	
	unsigned int cnt = 4;
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, uint32_t(_param_size)));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _betta1));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, w_decay));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, step_size));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, combined_unscale));

	size_t offset = 0;	

	float* p2p_param = p2p_params[device_id];	
	half* p2p_grad = p2p_grads[device_id];	
	float* p2p_exp_avg = p2p_exp_avgs[device_id];	

	int nvmeFd_param = -1;
	int nvmeFd_exp_avg = -1;
	int nvmeFd_grad = -1;
	

	//std::cout<<"read starts..."<<std::endl;
		
	nvmeFd_grad = open(grad_path.c_str(), O_RDWR | O_SYNC | O_DIRECT, 0644);
	ret = pread(nvmeFd_grad, (void*)p2p_grad, nbytes/2, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_param[device_id] != nullptr) {  write_param[device_id]->join(); write_param[device_id] = nullptr; }
	nvmeFd_param = open(param_path.c_str(), O_RDWR | O_SYNC  | O_DIRECT, 0644);
	ret = pread(nvmeFd_param, (void*)p2p_param, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg[device_id] != nullptr) {  write_exp_avg[device_id]->join(); write_exp_avg[device_id] = nullptr; }
	nvmeFd_exp_avg = open(exp_avg_path.c_str(), O_RDWR | O_SYNC  | O_DIRECT, 0644);
	ret = pread(nvmeFd_exp_avg, (void*)p2p_exp_avg, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

    //Launch the Kernel
	q.enqueueTask(krnl_adam, nullptr, nullptr);
	q.enqueueReadBuffer ( param16_pool[device_id], CL_FALSE, 0, nbytes/2, fp16_params_ptr);
	q.finish();

	(void)close(nvmeFd_grad);
	
	write_param[device_id] = new std::thread(write_thread, nvmeFd_param, p2p_param, nbytes, true);

	write_exp_avg[device_id] = new std::thread(write_thread, nvmeFd_exp_avg, p2p_exp_avg, nbytes, true);
}

void thread_work(
		std::string param_path,
		std::string exp_avg_path,
		std::string exp_avg_sq_path,
		std::string grad_path,
		size_t _param_size ,
		float combined_unscale,
		half* fp16_params_ptr,
		int device_id,
		float _alpha,
	    float _betta1,
	    float _betta2,
	    float _eps,
	    float _weight_decay,
		float _bias_correction1,
        float _bias_correction2
	)
{
	assert( int(_param_size) % (16) == 0);
	
	auto context = contexts[device_id];
	auto q = queues[device_id];

	auto krnl_adam = krnls[device_id];	

	int ret;

	size_t nbytes = _param_size * sizeof(float);
	
	float step_size = -1 * _alpha / _bias_correction1;
	float w_decay = -1 * _alpha * _weight_decay;

	cl_int err;
	
	unsigned int cnt = 5;
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, uint32_t(_param_size)));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _betta1));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _betta2));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _bias_correction2));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, _eps));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, w_decay));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, step_size));
	OCL_CHECK(err, err = krnl_adam.setArg(cnt++, combined_unscale));


	size_t offset = 0;	

	float* p2p_param = p2p_params[device_id];	
	half* p2p_grad = p2p_grads[device_id];	
	float* p2p_exp_avg = p2p_exp_avgs[device_id];	
	float* p2p_exp_avg_sq = p2p_exp_avg_sqs[device_id];	

	int nvmeFd_param = -1;
	int nvmeFd_exp_avg = -1;
	int nvmeFd_exp_avg_sq = -1;
	int nvmeFd_grad = -1;
	

	//std::cout<<"read starts..."<<std::endl;
		
	nvmeFd_grad = open(grad_path.c_str(), O_RDWR | O_SYNC | O_DIRECT, 0644);
	ret = pread(nvmeFd_grad, (void*)p2p_grad, nbytes/2, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_param[device_id] != nullptr) {  write_param[device_id]->join(); write_param[device_id] = nullptr; }
	nvmeFd_param = open(param_path.c_str(), O_RDWR | O_SYNC  | O_DIRECT, 0644);
	ret = pread(nvmeFd_param, (void*)p2p_param, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg[device_id] != nullptr) {  write_exp_avg[device_id]->join(); write_exp_avg[device_id] = nullptr; }
	nvmeFd_exp_avg = open(exp_avg_path.c_str(), O_RDWR | O_SYNC  | O_DIRECT, 0644);
	ret = pread(nvmeFd_exp_avg, (void*)p2p_exp_avg, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }

	if (write_exp_avg_sq[device_id] != nullptr) {  write_exp_avg_sq[device_id]->join(); write_exp_avg_sq[device_id] = nullptr ;}
	nvmeFd_exp_avg_sq = open(exp_avg_sq_path.c_str(), O_RDWR | O_SYNC | O_DIRECT, 0644);
	ret = pread(nvmeFd_exp_avg_sq, (void*)p2p_exp_avg_sq, nbytes, 0);
	if (ret == -1) { std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl; }
	
    //Launch the Kernel
	q.enqueueTask(krnl_adam, nullptr, nullptr);
	q.enqueueReadBuffer ( param16_pool[device_id], CL_FALSE, 0, nbytes/2, fp16_params_ptr);
	q.finish();

	(void)close(nvmeFd_grad);
	
	write_param[device_id] = new std::thread(write_thread, nvmeFd_param, p2p_param, nbytes, true);

	write_exp_avg[device_id] = new std::thread(write_thread, nvmeFd_exp_avg, p2p_exp_avg, nbytes, true);
	
	write_exp_avg_sq[device_id] = new std::thread(write_thread, nvmeFd_exp_avg_sq, p2p_exp_avg_sq, nbytes, true);
	
}

void Adam_Optimizer::Step_fpga_comp( 
				std::string param_path,
                std::string exp_avg_path,
                std::string exp_avg_sq_path,
                std::string grad_path,
				size_t _param_size ,
				float combined_unscale,
				half* fp16_params_ptr,
				int device_id,
				int largest_numel,
				float compression_ratio,
				int* grad_idx_ptr
				)
{
	//device_id = (MAX_DEVICE - 1) - device_id;
	int i;
	
	if (!(init[device_id])){	
		cl::Platform::get(&platforms[device_id]);
		cl::Platform platform;
		const std::string vendor_name = "Xilinx";
		for (i  = 0 ; i < platforms[device_id].size(); i++){
			platform = platforms[device_id][i];
			std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(nullptr);
			if (platformName == vendor_name){
				break;
			}
		}
		if (i == platforms[device_id].size()) {
			std::cout << "Error: Failed to find Xilinx platform" << std::endl;
			exit(EXIT_FAILURE);
		}
		platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices[device_id]);
		devices[device_id][0] = devices[device_id][device_id];
		devices[device_id].resize(1);
		cl_int err;
		char device_bdf[20];
        OCL_CHECK(err, err = devices[device_id][0].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
		std:: cout << "Device id: " << device_id <<"=>"<< device_bdf << std::endl;
		
		const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/topk_adam_fp16.xclbin";
		
		// Load xclbin 
		std::ifstream bin_file(xclbin_file_name, std::ifstream::binary);
		bin_file.seekg(0, bin_file.end);
		auto nb = bin_file.tellg();
		file_bufs[device_id] = new char [nb];
		bin_file.seekg(0, bin_file.beg);
		bin_file.read(file_bufs[device_id], nb);
		bin_file.close();
		std::cout << "Bin file read success " << std::endl;

		bins[device_id].push_back({file_bufs[device_id], nb});
		
		contexts[device_id] = cl::Context(devices[device_id]); 
		queues[device_id] = cl::CommandQueue(contexts[device_id], devices[device_id][0], 0, NULL);

		programs[device_id] = cl::Program(contexts[device_id], devices[device_id], bins[device_id]);
		
		krnls[device_id] = cl::Kernel(programs[device_id], "krnl_vadd");
		
		cl_mem_ext_ptr_t outExt = {0};
		outExt.flags = XCL_MEM_EXT_P2P_BUFFER;

		size_t nbytes = largest_numel * sizeof(float);
		size_t remainder = int( largest_numel * compression_ratio ) % 1024
		size_t comp_nbytes = ( (int( largest_numel * compression_ratio ) + 1024) -remainder ) * sizeof(float);

		OCL_CHECK(err, grad_idx_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, comp_nbytes, &outExt, &err));

		OCL_CHECK(err, grad_val_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, comp_nbytes/2, &outExt, &err));
		
		OCL_CHECK(err, grad_pool[device_id] =  cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2));

		OCL_CHECK(err, param16_pool[device_id] =  cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2));

		OCL_CHECK(err, param_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		OCL_CHECK(err, exp_avg_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		OCL_CHECK(err, exp_avg_sq_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));

		std::cout << "Buffer initilized success " << std::endl;

		p2p_params[device_id] = (float*)queues[device_id].enqueueMapBuffer(
											param_pool[device_id],						// buffer
											CL_FALSE,						// blocking call
											CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
											0,							// buffer offset
											nbytes,			// size in bytes
										    nullptr,          // waiting events vector
											nullptr,          // mapping event
											&err);
	

		p2p_exp_avgs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
	
		p2p_exp_avg_sqs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_sq_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
		
		p2p_grads_idx[device_id] = (int*)queues[device_id].enqueueMapBuffer(
									  grad_idx_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ | CL_MAP_WRITE,	//Indicates we will write
									  0,							// buffer offset
									  comp_nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		p2p_grads_val[device_id] = (half*)queues[device_id].enqueueMapBuffer(
									  grad_val_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ | CL_MAP_WRITE,	//Indicates we will write
									  0,							// buffer offset
									  comp_nbytes/2,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
		

		int cnt = 0;

		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_idx_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_val_pool[device_id]));
		cnt++;	// Param size

		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param16_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_sq_pool[device_id]));

		std::cout << "OpenCL initialize finished" << std::endl;
		init[device_id] = true;
	}

	threads.push_back(std::thread(thread_work_comp,
		param_path,
		exp_avg_path,
		exp_avg_sq_path,
		grad_path,
		_param_size ,
		combined_unscale,
		fp16_params_ptr,
		device_id,
		_alpha,
	    _betta1,
	    _betta2,
	    _eps,
	    _weight_decay,
		_bias_correction1,
        _bias_correction2,
		compression_ratio,
		grad_idx_ptr
		));
}

void Adam_Optimizer::Step_fpga_sgd_comp( 
				std::string param_path,
                std::string exp_avg_path,
                std::string grad_path,
				size_t _param_size ,
				float combined_unscale,
				half* fp16_params_ptr,
				int device_id,
				int largest_numel,
				float compression_ratio,
				int* grad_idx_ptr
				)
{
	//device_id = (MAX_DEVICE - 1) - device_id;
	int i;
	
	if (!(init[device_id])){	
		cl::Platform::get(&platforms[device_id]);
		cl::Platform platform;
		const std::string vendor_name = "Xilinx";
		for (i  = 0 ; i < platforms[device_id].size(); i++){
			platform = platforms[device_id][i];
			std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(nullptr);
			if (platformName == vendor_name){
				break;
			}
		}
		if (i == platforms[device_id].size()) {
			std::cout << "Error: Failed to find Xilinx platform" << std::endl;
			exit(EXIT_FAILURE);
		}
		platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices[device_id]);
		devices[device_id][0] = devices[device_id][device_id];
		devices[device_id].resize(1);
		cl_int err;
		char device_bdf[20];
        OCL_CHECK(err, err = devices[device_id][0].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
		std:: cout << "Device id: " << device_id <<"=>"<< device_bdf << std::endl;
		
		const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/topk_sgd.xclbin";
		
		// Load xclbin 
		std::ifstream bin_file(xclbin_file_name, std::ifstream::binary);
		bin_file.seekg(0, bin_file.end);
		auto nb = bin_file.tellg();
		file_bufs[device_id] = new char [nb];
		bin_file.seekg(0, bin_file.beg);
		bin_file.read(file_bufs[device_id], nb);
		bin_file.close();
		std::cout << "Bin file read success " << std::endl;

		bins[device_id].push_back({file_bufs[device_id], nb});
		
		contexts[device_id] = cl::Context(devices[device_id]); 
		queues[device_id] = cl::CommandQueue(contexts[device_id], devices[device_id][0], 0, NULL);

		programs[device_id] = cl::Program(contexts[device_id], devices[device_id], bins[device_id]);
		
		krnls[device_id] = cl::Kernel(programs[device_id], "krnl_vadd");
		
		cl_mem_ext_ptr_t outExt = {0};
		outExt.flags = XCL_MEM_EXT_P2P_BUFFER;

		size_t nbytes = largest_numel * sizeof(float);
		size_t comp_nbytes = ( (int( largest_numel * compression_ratio ) - 1) / 1024  + 1) * 1024 * sizeof(float);
		//size_t comp_nbytes = ( ( - 1) / 1024  + 1) * 1024 * sizeof(float);

		//OCL_CHECK(err, grad_idx_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, comp_nbytes ));
		OCL_CHECK(err, grad_idx_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, comp_nbytes, &outExt, &err));

		OCL_CHECK(err, grad_val_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, comp_nbytes/2, &outExt, &err));
		
		OCL_CHECK(err, grad_pool[device_id] =  cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2));

		OCL_CHECK(err, param16_pool[device_id] =  cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2));

		OCL_CHECK(err, param_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		OCL_CHECK(err, exp_avg_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));

		std::cout << "Buffer initilized success " << std::endl;

		p2p_params[device_id] = (float*)queues[device_id].enqueueMapBuffer(
											param_pool[device_id],						// buffer
											CL_FALSE,						// blocking call
											CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
											0,							// buffer offset
											nbytes,			// size in bytes
										    nullptr,          // waiting events vector
											nullptr,          // mapping event
											&err);
	

		p2p_exp_avgs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
	
		p2p_grads_idx[device_id] = (int*)queues[device_id].enqueueMapBuffer(
									  grad_idx_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ | CL_MAP_WRITE,	//Indicates we will write
									  0,							// buffer offset
									  comp_nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		p2p_grads_val[device_id] = (half*)queues[device_id].enqueueMapBuffer(
									  grad_val_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ | CL_MAP_WRITE,	//Indicates we will write
									  0,							// buffer offset
									  comp_nbytes/2,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
		

		int cnt = 0;

		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_idx_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_val_pool[device_id]));
		cnt++;	// Param size

		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param16_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_pool[device_id]));

		std::cout << "OpenCL initialize finished" << std::endl;
		init[device_id] = true;
	}

	threads.push_back(std::thread(thread_work_sgd_comp,
		param_path,
		exp_avg_path,
		grad_path,
		_param_size ,
		combined_unscale,
		fp16_params_ptr,
		device_id,
		_alpha,
	    _betta1,
	    _weight_decay,
		compression_ratio,
		grad_idx_ptr
		));
}



void Adam_Optimizer::Step_fpga_adagrad_comp( 
				std::string param_path,
                std::string exp_avg_sq_path,
                std::string grad_path,
				size_t _param_size ,
				float combined_unscale,
				half* fp16_params_ptr,
				int device_id,
				int largest_numel,
				float compression_ratio
				)
{
	//device_id = (MAX_DEVICE - 1) - device_id;
	int i;
	
	if (!(init[device_id])){	
		cl::Platform::get(&platforms[device_id]);
		cl::Platform platform;
		const std::string vendor_name = "Xilinx";
		for (i  = 0 ; i < platforms[device_id].size(); i++){
			platform = platforms[device_id][i];
			std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(nullptr);
			if (platformName == vendor_name){
				break;
			}
		}
		if (i == platforms[device_id].size()) {
			std::cout << "Error: Failed to find Xilinx platform" << std::endl;
			exit(EXIT_FAILURE);
		}
		platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices[device_id]);
		devices[device_id][0] = devices[device_id][device_id];
		devices[device_id].resize(1);
		cl_int err;
		char device_bdf[20];
        OCL_CHECK(err, err = devices[device_id][0].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
		std:: cout << "Device id: " << device_id <<"=>"<< device_bdf << std::endl;
		
		const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/topk_adagrad.xclbin";
		
		// Load xclbin 
		std::ifstream bin_file(xclbin_file_name, std::ifstream::binary);
		bin_file.seekg(0, bin_file.end);
		auto nb = bin_file.tellg();
		file_bufs[device_id] = new char [nb];
		bin_file.seekg(0, bin_file.beg);
		bin_file.read(file_bufs[device_id], nb);
		bin_file.close();
		std::cout << "Bin file read success " << std::endl;

		bins[device_id].push_back({file_bufs[device_id], nb});
		
		contexts[device_id] = cl::Context(devices[device_id]); 
		queues[device_id] = cl::CommandQueue(contexts[device_id], devices[device_id][0], 0, NULL);

		programs[device_id] = cl::Program(contexts[device_id], devices[device_id], bins[device_id]);
		
		krnls[device_id] = cl::Kernel(programs[device_id], "krnl_vadd");
		
		cl_mem_ext_ptr_t outExt = {0};
		outExt.flags = XCL_MEM_EXT_P2P_BUFFER;

		size_t nbytes = largest_numel * sizeof(float);
		size_t comp_nbytes = ( (int( largest_numel * compression_ratio ) - 1) / 1024  + 1) * 1024 * sizeof(float);

		//OCL_CHECK(err, grad_idx_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, comp_nbytes ));
		OCL_CHECK(err, grad_idx_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, comp_nbytes, &outExt, &err));

		OCL_CHECK(err, grad_val_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, comp_nbytes/2, &outExt, &err));
		
		OCL_CHECK(err, grad_pool[device_id] =  cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2));

		OCL_CHECK(err, param16_pool[device_id] =  cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2));

		OCL_CHECK(err, param_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		OCL_CHECK(err, exp_avg_sq_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));

		std::cout << "Buffer initilized success " << std::endl;

		p2p_params[device_id] = (float*)queues[device_id].enqueueMapBuffer(
											param_pool[device_id],						// buffer
											CL_FALSE,						// blocking call
											CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
											0,							// buffer offset
											nbytes,			// size in bytes
										    nullptr,          // waiting events vector
											nullptr,          // mapping event
											&err);
	

		p2p_exp_avg_sqs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_sq_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
		
		p2p_grads_idx[device_id] = (int*)queues[device_id].enqueueMapBuffer(
									  grad_idx_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ | CL_MAP_WRITE,	//Indicates we will write
									  0,							// buffer offset
									  comp_nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		p2p_grads_val[device_id] = (half*)queues[device_id].enqueueMapBuffer(
									  grad_val_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ | CL_MAP_WRITE,	//Indicates we will write
									  0,							// buffer offset
									  comp_nbytes/2,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
		

		int cnt = 0;

		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_idx_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_val_pool[device_id]));
		cnt++;	// Param size

		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param16_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_sq_pool[device_id]));

		std::cout << "OpenCL initialize finished" << std::endl;
		init[device_id] = true;
	}

	threads.push_back(std::thread(thread_work_adagrad_comp,
		param_path,
		exp_avg_sq_path,
		grad_path,
		_param_size ,
		combined_unscale,
		fp16_params_ptr,
		device_id,
		_alpha,
	    _eps,
	    _weight_decay,
		compression_ratio
		));
}

void Adam_Optimizer::Step_fpga_adagrad( 
				std::string param_path,
                std::string exp_avg_sq_path,
                std::string grad_path,
				size_t _param_size ,
				float combined_unscale,
				half* fp16_params_ptr,
				int device_id,
				int largest_numel
				)
{
	//device_id = (MAX_DEVICE - 1) - device_id;
	int i;
	if (!(init[device_id])){	
		cl::Platform::get(&platforms[device_id]);
		cl::Platform platform;
		const std::string vendor_name = "Xilinx";
		for (i  = 0 ; i < platforms[device_id].size(); i++){
			platform = platforms[device_id][i];
			std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(nullptr);
			if (platformName == vendor_name){
				break;
			}
		}
		if (i == platforms[device_id].size()) {
			std::cout << "Error: Failed to find Xilinx platform" << std::endl;
			exit(EXIT_FAILURE);
		}
		platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices[device_id]);
		devices[device_id][0] = devices[device_id][device_id];
		devices[device_id].resize(1);
		cl_int err;
		char device_bdf[20];
        OCL_CHECK(err, err = devices[device_id][0].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
		std:: cout << "Device id: " << device_id <<"=>"<< device_bdf << std::endl;
		
		const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/adagrad.xclbin";
		
		// Load xclbin 
		std::ifstream bin_file(xclbin_file_name, std::ifstream::binary);
		bin_file.seekg(0, bin_file.end);
		auto nb = bin_file.tellg();
		file_bufs[device_id] = new char [nb];
		bin_file.seekg(0, bin_file.beg);
		bin_file.read(file_bufs[device_id], nb);
		bin_file.close();

		bins[device_id].push_back({file_bufs[device_id], nb});
		
		contexts[device_id] = cl::Context(devices[device_id]); 
		queues[device_id] = cl::CommandQueue(contexts[device_id], devices[device_id][0], 0, NULL);

		programs[device_id] = cl::Program(contexts[device_id], devices[device_id], bins[device_id]);
		
		krnls[device_id] = cl::Kernel(programs[device_id], "krnl_vadd");
		
		cl_mem_ext_ptr_t outExt = {0};
		outExt.flags = XCL_MEM_EXT_P2P_BUFFER;

		size_t nbytes = largest_numel * sizeof(float);

		OCL_CHECK(err, param16_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2, nullptr, &err));

		OCL_CHECK(err, param_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		
		OCL_CHECK(err, exp_avg_sq_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));

		OCL_CHECK(err, grad_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes / 2, &outExt, &err));

		p2p_params[device_id] = (float*)queues[device_id].enqueueMapBuffer(
											param_pool[device_id],						// buffer
											CL_FALSE,						// blocking call
											CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
											0,							// buffer offset
											nbytes,			// size in bytes
										    nullptr,          // waiting events vector
											nullptr,          // mapping event
											&err);
	
		p2p_exp_avg_sqs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_sq_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		p2p_grads[device_id] = (half*)queues[device_id].enqueueMapBuffer(
									  grad_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes/2,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		int cnt = 0;
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param16_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_sq_pool[device_id]));

		init[device_id] = true;
	}

	threads.push_back(std::thread(thread_work_adagrad,
		param_path,
		exp_avg_sq_path,
		grad_path,
		_param_size ,
		combined_unscale,
		fp16_params_ptr,
		device_id,
		_alpha,
	    _eps,
	    _weight_decay
		));
}

void Adam_Optimizer::Step_fpga_sgd( 
				std::string param_path,
                std::string exp_avg_path,
                std::string grad_path,
				size_t _param_size ,
				float combined_unscale,
				half* fp16_params_ptr,
				int device_id,
				int largest_numel
				)
{
	//device_id = (MAX_DEVICE - 1) - device_id;

	int i;
	if (!(init[device_id])){	
			
		cl::Platform::get(&platforms[device_id]);
		cl::Platform platform;
		const std::string vendor_name = "Xilinx";
		for (i  = 0 ; i < platforms[device_id].size(); i++){
			platform = platforms[device_id][i];
			std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(nullptr);
			if (platformName == vendor_name){
				break;
			}
		}
		if (i == platforms[device_id].size()) {
			std::cout << "Error: Failed to find Xilinx platform" << std::endl;
			exit(EXIT_FAILURE);
		}
		platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices[device_id]);
		devices[device_id][0] = devices[device_id][device_id];
		devices[device_id].resize(1);
		cl_int err;
		char device_bdf[20];
        OCL_CHECK(err, err = devices[device_id][0].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
		std:: cout << "Device id: " << device_id <<"=>"<< device_bdf << std::endl;
		
		const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/sgd.xclbin";
		
		// Load xclbin 
		std::ifstream bin_file(xclbin_file_name, std::ifstream::binary);
		bin_file.seekg(0, bin_file.end);
		auto nb = bin_file.tellg();
		file_bufs[device_id] = new char [nb];
		bin_file.seekg(0, bin_file.beg);
		bin_file.read(file_bufs[device_id], nb);
		bin_file.close();

		bins[device_id].push_back({file_bufs[device_id], nb});
		
		contexts[device_id] = cl::Context(devices[device_id]); 
		queues[device_id] = cl::CommandQueue(contexts[device_id], devices[device_id][0], 0, NULL);

		programs[device_id] = cl::Program(contexts[device_id], devices[device_id], bins[device_id]);
		
		krnls[device_id] = cl::Kernel(programs[device_id], "krnl_vadd");
		
		cl_mem_ext_ptr_t outExt = {0};
		outExt.flags = XCL_MEM_EXT_P2P_BUFFER;

		size_t nbytes = largest_numel * sizeof(float);

		OCL_CHECK(err, param16_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2, nullptr, &err));

		OCL_CHECK(err, param_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		
		OCL_CHECK(err, exp_avg_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		
		OCL_CHECK(err, grad_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes / 2, &outExt, &err));


		p2p_params[device_id] = (float*)queues[device_id].enqueueMapBuffer(
											param_pool[device_id],						// buffer
											CL_FALSE,						// blocking call
											CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
											0,							// buffer offset
											nbytes,			// size in bytes
										    nullptr,          // waiting events vector
											nullptr,          // mapping event
											&err);
	

		p2p_exp_avgs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
	

		p2p_grads[device_id] = (half*)queues[device_id].enqueueMapBuffer(
									  grad_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes/2,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		int cnt = 0;
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param16_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_pool[device_id]));

		init[device_id] = true;
	}

	threads.push_back(std::thread(thread_work_sgd,
		param_path,
		exp_avg_path,
		grad_path,
		_param_size ,
		combined_unscale,
		fp16_params_ptr,
		device_id,
		_alpha,
	    _betta1,
	    _weight_decay
		));
}


void Adam_Optimizer::Step_fpga( 
				std::string param_path,
                std::string exp_avg_path,
                std::string exp_avg_sq_path,
                std::string grad_path,
				size_t _param_size ,
				float combined_unscale,
				half* fp16_params_ptr,
				int device_id,
				int largest_numel
				)
{
	//device_id = (MAX_DEVICE - 1) - device_id;

	int i;
	if (!(init[device_id])){	
			
		cl::Platform::get(&platforms[device_id]);
		cl::Platform platform;
		const std::string vendor_name = "Xilinx";
		for (i  = 0 ; i < platforms[device_id].size(); i++){
			platform = platforms[device_id][i];
			std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(nullptr);
			if (platformName == vendor_name){
				break;
			}
		}
		if (i == platforms[device_id].size()) {
			std::cout << "Error: Failed to find Xilinx platform" << std::endl;
			exit(EXIT_FAILURE);
		}
		platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices[device_id]);
		devices[device_id][0] = devices[device_id][device_id];
		devices[device_id].resize(1);
		cl_int err;
		char device_bdf[20];
        OCL_CHECK(err, err = devices[device_id][0].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
		std:: cout << "Device id: " << device_id <<"=>"<< device_bdf << std::endl;
		
		//const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/adam_fp16.xclbin";
		const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/slow_adam.xclbin";
		//const char* xclbin_file_name = "/mnt/home/hsjang0918/bins/adam.xclbin";
		
		// Load xclbin 
		std::ifstream bin_file(xclbin_file_name, std::ifstream::binary);
		bin_file.seekg(0, bin_file.end);
		auto nb = bin_file.tellg();
		file_bufs[device_id] = new char [nb];
		bin_file.seekg(0, bin_file.beg);
		bin_file.read(file_bufs[device_id], nb);
		bin_file.close();

		bins[device_id].push_back({file_bufs[device_id], nb});
		
		contexts[device_id] = cl::Context(devices[device_id]); 
		queues[device_id] = cl::CommandQueue(contexts[device_id], devices[device_id][0], 0, NULL);

		programs[device_id] = cl::Program(contexts[device_id], devices[device_id], bins[device_id]);
		
		krnls[device_id] = cl::Kernel(programs[device_id], "krnl_vadd");
		
		cl_mem_ext_ptr_t outExt = {0};
		outExt.flags = XCL_MEM_EXT_P2P_BUFFER;

		size_t nbytes = largest_numel * sizeof(float);

		OCL_CHECK(err, param16_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE, nbytes/2, nullptr, &err));

		OCL_CHECK(err, param_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		
		OCL_CHECK(err, exp_avg_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));
		
		OCL_CHECK(err, exp_avg_sq_pool[device_id] = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes, &outExt, &err));

		OCL_CHECK(err, grad_pool[device_id]  = cl::Buffer(contexts[device_id], CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, nbytes / 2, &outExt, &err));


		p2p_params[device_id] = (float*)queues[device_id].enqueueMapBuffer(
											param_pool[device_id],						// buffer
											CL_FALSE,						// blocking call
											CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
											0,							// buffer offset
											nbytes,			// size in bytes
										    nullptr,          // waiting events vector
											nullptr,          // mapping event
											&err);
	

		p2p_exp_avgs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);
	
		p2p_exp_avg_sqs[device_id] = (float*)queues[device_id].enqueueMapBuffer(
									  exp_avg_sq_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_WRITE | CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		p2p_grads[device_id] = (half*)queues[device_id].enqueueMapBuffer(
									  grad_pool[device_id],						// buffer
									  CL_FALSE,						// blocking call
									  CL_MAP_READ,	//Indicates we will write
									  0,							// buffer offset
									  nbytes/2,			// size in bytes
									  nullptr, 
									  nullptr,
									  &err);

		int cnt = 0;
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, grad_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param16_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, param_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_pool[device_id]));
		OCL_CHECK(err, err = krnls[device_id].setArg(cnt++, exp_avg_sq_pool[device_id]));

		init[device_id] = true;
	}

	threads.push_back(std::thread(thread_work,
		param_path,
		exp_avg_path,
		exp_avg_sq_path,
		grad_path,
		_param_size ,
		combined_unscale,
		fp16_params_ptr,
		device_id,
		_alpha,
	    _betta1,
	    _betta2,
	    _eps,
	    _weight_decay,
		_bias_correction1,
        _bias_correction2
		));
}



void Adam_Optimizer::Step_gpu(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
	float betta1_minus1 = 1 - _betta1;
	float betta2_minus1 = 1 - _betta2;

	float step_size = -1 * _alpha / _bias_correction1;
	float w_decay = -1 * _alpha * _weight_decay;

	for (size_t offset = 0; offset < _param_size; offset += TILE) {
			
        size_t copy_size = TILE;
        if ((offset + TILE) > _param_size) copy_size = _param_size - offset;
		
		adam_gpu(_params + offset, grads + offset,
				_exp_avg + offset, _exp_avg_sq + offset, 
				
				copy_size,

				step_size, w_decay,
				_betta1, _betta2,
				_eps,
				_bias_correction1, _bias_correction2);
	}
}

void Adam_Optimizer::Step_SGD_cpu(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
	//std::cout << "is half?" << half_precision << std::endl;

    size_t rounded_size = 0;
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float step_size = -1 * _alpha;
        float w_decay = _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = grads[k];
                float param = _params[k];
                float momentum = _exp_avg[k];
                //float variance = _exp_avg_sq[k];
				
                grad = param * w_decay + grad;
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                param = momentum * step_size + param;
                _params[k] = param;
                _exp_avg[k] = momentum;
            }
        }

    }
}


void Adam_Optimizer::Step_cpu(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;
        if (half_precision) {
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#endif // ENABLE_CUDA

#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif // ENABLE_CUDA
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }

#if defined(__ENABLE_CUDA__)
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);

                _buf_index = !_buf_index;
            }
#endif // ENABLE_CUDA
        }

    }
}

void Adam_Optimizer::Step_1(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;
        if (half_precision) {
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#endif
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
#if defined(__ENABLE_CUDA__)
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);

                _buf_index = !_buf_index;
            }
#endif
        }
    }
}

void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}


int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

	if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

int ds_adagrad_step_fpga(int optimizer_id,
                 size_t step,
                 float lr,
                 float epsilon,
                 float weight_decay,
				 float combined_unscale,
				 std::string param_path,
				 std::string exp_avg_sq_path,
				 std::string grad_path,
				 size_t _param_size, 
				 torch::Tensor& fp16_params,
				 int device_id,
				 int largest_numel,
				 float compression_ratio,
				 torch::Tensor& grad_idx
				 )
{
    //auto fp32_params_c = fp32_params.contiguous();
	//float* fp32_params_ptr = (float*)fp32_params_c.data_ptr();
    
	auto fp16_params_c = fp16_params.contiguous();
	half* fp16_params_ptr = (half*)fp16_params_c.data_ptr();

	std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    
	//opt->IncrementStep(step, beta1, beta2);
	opt->IncrementStep_(step);
	//opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->update_state(lr, epsilon, weight_decay, false);
	
	if (compression_ratio < 0.5){
		int* grad_idx_ptr = (int*)grad_idx.data_ptr();
		opt->Step_fpga_adagrad_comp(
				param_path,
                exp_avg_sq_path,
                grad_path,
                _param_size,
				combined_unscale,
				fp16_params_ptr,
				device_id,
				largest_numel,
				compression_ratio
				);
	}else{
		opt->Step_fpga_adagrad(
				param_path,
                exp_avg_sq_path,
                grad_path,
                _param_size,
				combined_unscale,
				fp16_params_ptr,
				device_id,
				largest_numel
				);
	}
    return 0;
}

int ds_sgd_step_fpga(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float weight_decay,
				 float combined_unscale,
				 std::string param_path,
				 std::string exp_avg_path,
				 std::string grad_path,
				 size_t _param_size, 
				 torch::Tensor& fp16_params,
				 int device_id,
				 int largest_numel,
				 float compression_ratio,
				 torch::Tensor& grad_idx
				 )
{
    //auto fp32_params_c = fp32_params.contiguous();
	//float* fp32_params_ptr = (float*)fp32_params_c.data_ptr();
    
	auto fp16_params_c = fp16_params.contiguous();
	half* fp16_params_ptr = (half*)fp16_params_c.data_ptr();

	std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep_( step, beta1 );
    opt->update_state(lr, 0.0, weight_decay, false);
	
	if (compression_ratio < 0.5){
		int* grad_idx_ptr = (int*)grad_idx.data_ptr();
		opt->Step_fpga_sgd_comp(
				param_path,
                exp_avg_path,
                grad_path,
                _param_size,
				combined_unscale,
				fp16_params_ptr,
				device_id,
				largest_numel,
				compression_ratio,
				grad_idx_ptr
				);
	}else{
		opt->Step_fpga_sgd(
				param_path,
                exp_avg_path,
                grad_path,
                _param_size,
				combined_unscale,
				fp16_params_ptr,
				device_id,
				largest_numel
				);
	}
    return 0;
}


int ds_adam_step_fpga(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction, 
				 float combined_unscale,
				 std::string param_path,
				 std::string exp_avg_path,
				 std::string exp_avg_sq_path,
				 std::string grad_path,
				 size_t _param_size, 
				 //torch::Tensor& fp32_params,
				 torch::Tensor& fp16_params,
				 int device_id,
				 int largest_numel,
				 float compression_ratio,
				 torch::Tensor& grad_idx
				 )
{
    //auto fp32_params_c = fp32_params.contiguous();
	//float* fp32_params_ptr = (float*)fp32_params_c.data_ptr();
    
	auto fp16_params_c = fp16_params.contiguous();
	half* fp16_params_ptr = (half*)fp16_params_c.data_ptr();

	std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
	
	if (compression_ratio < 0.5){
		int* grad_idx_ptr = (int*)grad_idx.data_ptr();
		opt->Step_fpga_comp(
				param_path,
                exp_avg_path,
                exp_avg_sq_path,
                grad_path,
                _param_size,
				combined_unscale,
				fp16_params_ptr,
				device_id,
				largest_numel,
				compression_ratio,
				grad_idx_ptr
				);
	}else{
		opt->Step_fpga(
				param_path,
                exp_avg_path,
                exp_avg_sq_path,
                grad_path,
                _param_size,
				combined_unscale,
				fp16_params_ptr,
				device_id,
				largest_numel
				);
	}
    return 0;
}

int ds_sgd_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep_(step, beta);
    opt->update_state(lr, 0.0, weight_decay, false);
	
    opt->Step_SGD_cpu(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                nullptr,
                params_c.numel(),
                nullptr,
                (params.options().dtype() == at::kHalf));

#if defined(__ENABLE_CUDA__)
    opt->SynchronizeStreams();
#endif

    return 0;
}


int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
	
#if ACC_TYPE == DS_CPU
    opt->Step_8(params_ptr,
#elif ACC_TYPE == CPU 
    opt->Step_cpu(params_ptr,
#elif ACC_TYPE == GPU
    opt->Step_gpu(params_ptr,
#endif
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                nullptr,
                (params.options().dtype() == at::kHalf));

#if defined(__ENABLE_CUDA__)
    opt->SynchronizeStreams();
#endif
    return 0;
}

int ds_adam_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& gpu_params)
{
#if defined(__ENABLE_CUDA__)
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    ds_half_precision_t* gpu_params_ptr = (ds_half_precision_t*)gpu_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                gpu_params_ptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return 0;
}

int destroy_adam_optimizer(int optimizer_id)
{
	assert(false);	
    s_optimizers.erase(optimizer_id);
    
    return 0;
}

void create_cl_buffer(){
}
	

//void finalize_cl_buf() 
//std::cout << " vadd finalized " << optimizer_id << std::endl;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
	m.def("sgd_update", &ds_sgd_step, "SmartInfinity SGD update (C++)");
    m.def("adam_update_copy",
          &ds_adam_step_plus_copy,
          "DeepSpeed CPU Adam update and param copy (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
    
	m.def("create_cl_buf", &create_cl_buffer, "Create buffer for OpenCL");
    
	m.def("adam_update_fpga", &ds_adam_step_fpga, "FPGA Adam update (C++)");
	m.def("adagrad_update_fpga", &ds_adagrad_step_fpga, "FPGA Adagrad update (C++)");
	m.def("sgd_update_fpga", &ds_sgd_step_fpga, "FPGA sgd update (C++)");
	
	m.def("sync_thread", &sync_thread, "FPGA Threads Sync (C++)");
}
