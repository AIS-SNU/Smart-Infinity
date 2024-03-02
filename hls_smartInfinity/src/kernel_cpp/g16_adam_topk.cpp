/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Demonstrate Vector Add in OpenCL
//

#include "hls_vector.h"
#include <cmath>
#include "hls_half.h" 
#define VEC_SIZE 16
#define D_VEC_SIZE 2* VEC_SIZE
#define DATA_SIZE 4096

#define TOPK 1

typedef unsigned int uint;

typedef hls::vector<float, VEC_SIZE> vec; // 512 bits
typedef hls::vector<uint, VEC_SIZE> ivec; // 512 bits
typedef hls::vector<half, VEC_SIZE> hvec; // 256 bits
typedef hls::vector<half, D_VEC_SIZE> dhvec; // 256 bits

static void initialize( half* grad16, uint n_elements )
{
	dhvec z = 0.0f;

	uint iteration = n_elements / VEC_SIZE;
	vadd_pipeline: for (uint i = 0 ; i < iteration ; i += DATA_SIZE)
    {
		#pragma HLS PIPELINE rewind
		uint size = DATA_SIZE;
		//boundary check
		if (i + size > iteration) size = iteration - i;

		init_grad: for ( uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			for (uint y = 0 ; y < VEC_SIZE;  ++y)
			{
			#pragma HLS UNROLL
				grad16[(i+x) * VEC_SIZE + y] = z[y];
			}
		}
    }
}

void decompressor( ivec* grad_idx, hvec* grad_val, uint n_elements_compressed, half* grad16 )
{
	ivec g_idx[DATA_SIZE];
	hvec g_val[DATA_SIZE];

	uint iteration = n_elements_compressed / VEC_SIZE;

	decompressor : for ( uint i = 0 ; i < iteration ; i+= DATA_SIZE )
	{
		#pragma HLS PIPELINE rewind
		uint size = DATA_SIZE;
		//boundary check
		if (i + size > iteration) size = iteration - i;

		read_grad_idx: for (uint x = 0 ; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			g_idx[x] = grad_idx[i + x];		
		}
		read_grad_val: for (uint x = 0 ; x < size ; ++x)
		{
			#pragma HLS PIPELINE II=1
			g_val[x] = grad_val[i + x];		
		}
		write_grad: for (uint x = 0 ; x < size; ++x)
		{
			inner_write_grad: for (uint y = 0 ; y < VEC_SIZE; ++y)
			{
				#pragma HLS PIPELINE II=1
				grad16[ g_idx[x][y] ] = g_val[x][y];		
			}
		}
	}
}

void adam (	half* grad16, dhvec* param16, vec* param, vec* exp_avg, vec* exp_avg_sq, uint n_elements, 
			float betta1, float betta2, float bias_correction2, float eps, float w_decay, float step_size, float combined_unscale )
{
	vec g[DATA_SIZE];
	vec p[DATA_SIZE];
	vec v[DATA_SIZE];
	vec m[DATA_SIZE];
	
	float _betta1 = betta1;
	float _betta2 = betta2;
	float _betta1_minus1 = (1.0f - _betta1);
	float _betta2_minus1 = (1.0f - _betta2);
	float _bias_correction2 = bias_correction2;
	float _eps = eps;
	float _w_decay_plus1 = (1.0f + w_decay);
	float _step_size = step_size;

	float _combined_unscale = combined_unscale;

	uint iteration = n_elements / VEC_SIZE;
	vadd_pipeline: for (uint i = 0 ; i < iteration ; i += DATA_SIZE)
	{
		#pragma HLS PIPELINE rewind
		uint size = DATA_SIZE;
		//boundary check
		if (i + size > iteration) size = iteration - i;
		
		uint size_h = size / 2;
		uint i_h = i / 2;

		read_g: for ( uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			inner_read_g: for (uint y = 0 ; y < VEC_SIZE;  ++y)
			{
			#pragma HLS UNROLL
				g[ x ][y ] = grad16[ (i + x) * VEC_SIZE + y] * _combined_unscale; // BRAM bitwidth is limited...
			}
		}
		read_v: for (uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			v[x] =  g[x] * g[x] * _betta2_minus1 + exp_avg_sq[i + x] * _betta2;
		}
		write_v: for (uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			exp_avg_sq[i + x] = v[x];
		}
		read_m: for (uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			m[x] = g[x] * _betta1_minus1 + exp_avg[i + x] * _betta1;
		}
		write_m: for (uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			exp_avg[i + x] = m[x];
		}
		read_p: for (uint x = 0 ; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			inner_read_p: for (uint y = 0 ; y < VEC_SIZE; ++y)
			{
			#pragma HLS UNROLL
				p[x][y] = _w_decay_plus1 * param[i + x][y] + ( m[x][y] / ( sqrtf(v[x][y])*_bias_correction2+_eps) ) * _step_size;
			}
		}
		write_p: for ( uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			param[i + x] = p[x];
		}
		float_to_half: for ( uint x = 0; x < size_h; ++x)
		{
			#pragma HLS PIPELINE II=1
			inner_f_to_h1:for (uint y = 0 ; y < VEC_SIZE; ++y)
			{
			#pragma HLS UNROLL
				param16[(i_h + x)][y] = p[2*x][y];
			}
			inner_f_to_h2:for (uint y = 0 ; y < VEC_SIZE; ++y)
			{
			#pragma HLS UNROLL
				param16[(i_h + x)][VEC_SIZE + y] = p[2*x + 1][y];
			}
		}
	}
}


extern "C"{
	void krnl_vadd(
#if TOPK
				ivec* grad_idx,
				hvec* grad_val,
				uint n_elements_compressed,
#endif
				half* grad16,
				dhvec* param16,
				vec* param,
				vec* exp_avg,
				vec* exp_avg_sq,
				uint  n_elements,
				float betta1,
				float betta2,
				float bias_correction2,
				float eps,
				float w_decay,
				float step_size,
				float combined_unscale
			)
	{
	#pragma HLS interface m_axi port=grad16 offset=slave bundle=half256 max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS interface m_axi port=param16 offset=slave bundle=half512 max_write_burst_length=256

	#pragma HLS interface m_axi port=param offset=slave bundle=single max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS interface m_axi port=exp_avg offset=slave bundle=single max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS interface m_axi port=exp_avg_sq offset=slave bundle=single max_read_burst_length=256  max_write_burst_length=256

	#pragma HLS interface m_axi port=grad_idx offset=slave bundle=single_comp max_read_burst_length=256 
	#pragma HLS interface m_axi port=grad_val offset=slave bundle=half_comp max_read_burst_length=256 


#if TOPK
		initialize(grad16, n_elements);
		decompressor( grad_idx, grad_val, n_elements_compressed, grad16 );
#endif
		adam ( grad16, param16, param, exp_avg, exp_avg_sq,  n_elements, 
					betta1, betta2, bias_correction2, eps, w_decay, step_size, combined_unscale );
	}
}
