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

typedef unsigned int uint;

typedef hls::vector<float, VEC_SIZE> vec; // 512 bits
typedef hls::vector<half, D_VEC_SIZE> dhvec; // 256 bits


void sgd (	dhvec* grad16, dhvec* param16, vec* param, vec* exp_avg, uint n_elements, 
			float betta, float w_decay, float step_size, float combined_unscale )
{
	vec g[DATA_SIZE];
	vec p[DATA_SIZE];
	vec m[DATA_SIZE];

	float _betta = betta;
	float _betta_minus1 = (1.0f - _betta);
	float _step_size = step_size;
	float _w_decay = w_decay;
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
		
		read_p: for (uint x = 0 ; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			p[x] = param[i + x];
		}

		read_g: for ( uint x = 0; x < size_h; ++x)
		{
			uint a = 2 * x;
			uint b = i_h + x;
			#pragma HLS PIPELINE II=1
			inner_read_g1: for (uint y = 0 ; y < VEC_SIZE;  ++y)
			{
			#pragma HLS UNROLL
				g[ a ][y ] = grad16[ b ][y] * _combined_unscale;
			}
			inner_read_g2: for (uint y = 0 ; y < VEC_SIZE;  ++y)
			{
			#pragma HLS UNROLL
				g[ a  + 1 ][y ] = grad16[ b ][VEC_SIZE + y] * _combined_unscale;
			}
		}


		read_m: for (uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			g[x] = g[x] + _w_decay * p[x];
			m[x] = _betta_minus1 * g[x] +  ( _betta )* exp_avg[ i + x];
		}

		write_m: for (uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			exp_avg[i + x] = m[x];
		}
		
		compute_p: for ( uint x = 0; x < size; ++x)
		{
			#pragma HLS PIPELINE II=1
			p[x] = p[x] + _step_size * m[x];
			param[i + x] = p[x];
		}

		float_to_half: for ( uint x = 0; x < size_h; ++x)
		{
			uint a = 2* x;
			uint b = i_h + x;
			#pragma HLS PIPELINE II=1
			inner_f_to_h1:for (uint y = 0 ; y < VEC_SIZE; ++y)
			{
			#pragma HLS UNROLL
				param16[b][y] = p[ a ][y];
			}
			inner_f_to_h2:for (uint y = 0 ; y < VEC_SIZE; ++y)
			{
			#pragma HLS UNROLL
				param16[b][VEC_SIZE + y] = p[ a + 1][y];
			}
		}
	}
}


extern "C"{
	void krnl_vadd(
				dhvec* grad16,
				dhvec* param16,
				vec* param,
				vec* exp_avg,
				uint  n_elements,
				float betta,
				float w_decay,
				float step_size,
				float combined_unscale
			)
	{
	#pragma HLS interface m_axi port=grad16 offset=slave bundle=half max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS interface m_axi port=param16 offset=slave bundle=half max_write_burst_length=256

	#pragma HLS interface m_axi port=param offset=slave bundle=single max_read_burst_length=256 max_write_burst_length=256
	#pragma HLS interface m_axi port=exp_avg offset=slave bundle=single max_read_burst_length=256  max_write_burst_length=256

	sgd ( grad16, param16, param, exp_avg,  n_elements, 
				   betta, w_decay, step_size, combined_unscale );
	}
}
