# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from cpuinfo import get_cpu_info
from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le
from deepspeed.ops.op_builder import CPUAdamBuilder

#from deepspeed.runtime.utils import see_memory_usage
#from deepspeed.runtime.swap_tensor.utils import get_sized_buffers, get_sized_buffer


class DeepSpeedCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True,
                 fp32_optimizer_states=True,
                 opt_type = 0):
        """Fast vectorized implementation of two variations of Adam optimizer on CPU:

        * Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        * AdamW: Fixing Weight Decay Regularization in Adam (https://arxiv.org/abs/1711.05101)

        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
        In order to apply this optimizer, the model requires to have its master parameter (in FP32)
        reside on the CPU memory.

        To train on a heterogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
        the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
        with minimal impact on training throughput. DeepSpeedCPUAdam plays an important role to minimize
        the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
        (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.

        For calling step function, there are two options available: (1) update optimizer's states and (2) update
        optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
        option can bring 30% higher throughput than the doing the copy separately using option one.


        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.


        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: select between Adam and AdamW implementations (default: AdamW)
            full_precision_optimizer_states: creates momentum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """
        self.opt_type = opt_type
        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)

        super(DeepSpeedCPUAdam, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() if "vendor_id_raw" in cpu_info else "unknown"
        if "amd" in self.cpu_vendor:
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group['params']):
                    if p.dtype == torch.half:
                        logger.warning("FP16 params for CPUAdam may not work on AMD CPUs")
                        break
                else:
                    continue
                break

        self.opt_id = DeepSpeedCPUAdam.optimizer_id
        if DeepSpeedCPUAdam.optimizer_id > 0:
            raise NotImplementedError

        DeepSpeedCPUAdam.optimizer_id = DeepSpeedCPUAdam.optimizer_id + 1
        
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.ds_opt_adam = CPUAdamBuilder().load()

        self.ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode,
                                     should_log_le("info"))

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        self.ds_opt_adam.destroy_adam(self.opt_id)
        print("<< opt_id : ", self.opt_id, "deleted")

    def __setstate__(self, state):
        super(DeepSpeedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        # converting the fp16 params to a group of parameter
        if type(fp16_param_groups) is list:
            if type(fp16_param_groups[0]) is not list:
                fp16_param_groups = [fp16_param_groups]
        elif fp16_param_groups is not None:
            fp16_param_groups = [[fp16_param_groups]]
    
        #print("group_id: ", self.param_groups )
        for group_id, group in enumerate(self.param_groups):
            #print("group_id: ", group_id )
            for param_id, p in enumerate(group['params']):
                #print("param_id: ", group_id )
                if p.grad is None:
                    #print("No grad!")
                    continue
                #print("Has grad!")

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    if self.opt_type == 0:
                        state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    elif self.opt_type == 1:
                        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    elif self.opt_type == 2:
                        state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                        pass # Momuentum SGD
                    else:
                        raise NotImplementedError
                        
                    
                    # Initilize buffer

                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                if fp16_param_groups is not None:
                    raise NotImplementedError
                    self.ds_opt_adam.adam_update_copy(self.opt_id, state['step'], group['lr'], beta1, beta2,
                                                      group['eps'], group['weight_decay'], group['bias_correction'],
                                                      p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'],
                                                      fp16_param_groups[group_id][param_id].data)
                else:
                    if self.opt_type == 0:
                        self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                                 group['weight_decay'], group['bias_correction'], 
                                                 p.data, p.grad.data,
                                                state['exp_avg'], state['exp_avg_sq'])
                    elif self.opt_type == 1:
                        pass
                    elif self.opt_type == 2:
                        self.ds_opt_adam.sgd_update(self.opt_id, state['step'], group['lr'], beta1,
                                                 group['weight_decay'], 
                                                 p.data, p.grad.data,
                                                state['exp_avg'])
                    else:
                        raise NotImplementedError
        return loss

    def _io_aligned_numel(self, numel, optimizer_swapper):
        remainder = numel % optimizer_swapper.numel_alignment
        return numel if remainder == 0 else (numel + optimizer_swapper.numel_alignment - remainder)

    def sync_thread(self):
        self.ds_opt_adam.sync_thread();

    def topk(self, tensor, top_size):
        pass


    @torch.no_grad()
    def step_with_fpga(self, device_id, optimizer_swapper, combined_unscale, largest_numel, compression_ratio = 1. ):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            subgroup_id  :  For prefetch, offload optimizer states
            combined_unscale: 1. / Combined grad norm
        """
        # intended device for step
        device = torch.device('cpu')
        
        #see_memory_usage(f'Before prepare optimizer sub group {sub_group_id}', force=False)
        for group_id, group in enumerate(self.param_groups):
            for param_id, (p16, p32) in enumerate(group['params']):
                
                assert p16.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
                assert p32.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."
                
                swap_info = optimizer_swapper.swap_params_info.get(id(p32), None)
                if swap_info is None:
                    print("swap_info is None!!")
                    raise NotImplementedError
                assert(swap_info.has_gradients)
            
                aligned_numel = self._io_aligned_numel(swap_info.numel(), optimizer_swapper)

                #print(swap_info.swap_paths) 
                param_path = swap_info.swap_paths[0]
                if self.opt_type == 0:
                    exp_avg_path = swap_info.swap_paths[1]
                    exp_avg_sq_path = swap_info.swap_paths[2]
                elif self.opt_type == 1:
                    exp_avg_sq_path = swap_info.swap_paths[1]
                elif self.opt_type == 2: # SGD
                    exp_avg_path = swap_info.swap_paths[1]
                else:
                    raise NotImplementedError

                idx = torch.Tensor(); # Not Used!

                if compression_ratio < 0.5 :
                    assert(len(swap_info.swapped_gradients) == 2)
                    grad_path = swap_info.swapped_gradients[0].path[:-1]
                else:
                    grad_path = swap_info.swapped_gradients[0].path

                state = self.state[p32]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p32.dtype

                    if self.opt_type == 0:# Adadm
                        state['exp_avg'] = torch.zeros_like(p32.data, dtype=state_dtype, device=device)
                        state['exp_avg_sq'] = torch.zeros_like(p32.data, dtype=state_dtype, device=device)
                    if self.opt_type == 1:# Adagrad
                        state['exp_avg_sq'] = torch.zeros_like(p32.data, dtype=state_dtype, device=device)
                    elif self.opt_type == 2:# Momentum SGD
                        state['exp_avg'] = torch.zeros_like(p32.data, dtype=state_dtype, device=device)
                    else:
                        raise NotImplementedError


                state['step'] += 1
                beta1, beta2 = group['betas']

                assert(p32 is not None)
                if self.opt_type == 0:
                    self.ds_opt_adam.adam_update_fpga(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                             group['weight_decay'], group['bias_correction'], combined_unscale,
                                            param_path, exp_avg_path, exp_avg_sq_path, grad_path, aligned_numel, 
                                            p16.data, device_id, largest_numel, compression_ratio, idx.data)
                elif self.opt_type == 1:
                    self.ds_opt_adam.adagrad_update_fpga(self.opt_id, state['step'], group['lr'], group['eps'],
                                             group['weight_decay'], combined_unscale,
                                            param_path, exp_avg_sq_path, grad_path, aligned_numel, 
                                            p16.data, device_id, largest_numel, compression_ratio, idx.data)
                elif self.opt_type == 2:
                    self.ds_opt_adam.sgd_update_fpga(self.opt_id, state['step'], group['lr'], beta1,
                                             group['weight_decay'], combined_unscale,
                                            param_path, exp_avg_path, grad_path, aligned_numel, 
                                            p16.data, device_id, largest_numel, compression_ratio, idx.data)

                else:
                    raise NotImplementedError


        #unflatten fp16 parameter subgroup
        #self._unflatten_partitioned_parameters(sub_group_id)

        #see_memory_usage(f'After release optimizer sub group {sub_group_id}', force=False)
