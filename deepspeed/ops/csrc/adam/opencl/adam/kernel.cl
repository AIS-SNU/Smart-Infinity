
__kernel void vadd(__global float *params,  
					__global float *grads, 
					__global float *exp_avg, 
					__global float *exp_avg_sq, 
					
					unsigned long param_size,

					float step_size,  
					float w_decay, 

					float _betta1, 
					float _betta2,

					float _eps,

					float _bias_correction1,
					float _bias_correction2
					) {
	
	const unsigned int k = get_global_id(0);
	
	if ( k < param_size){
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
}
