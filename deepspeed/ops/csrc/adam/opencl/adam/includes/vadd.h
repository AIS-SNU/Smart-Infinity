#pragma once
#include <stdlib.h>

void adam_initialize();
void adam_create_buf(size_t M);
void adam_finalize();

void adam_cpu(float* _params,
		float* grads,
		float* _exp_avg,
		float* _exp_avg_sq,
		size_t _param_size);

void adam_gpu(float* _params,
		const float* grads,
		float* _exp_avg,
		float* _exp_avg_sq,
		size_t _param_size,
		float step_size, float w_decay,
		
		float _betta1 = 0.9, float _betta2 = 0.999,
		float _eps = 1e-8, 
		float _bias_correction1 = 1.0f, float _bias_correction2 = 1.0f
		);

void check_adam(
		float* params,
		float* grads,
		float* exp_avg,
		float* exp_avg_sq,
		float* _params,
		float* _exp_avg,
		float* _exp_avg_sq,
		size_t _param_size
		);


