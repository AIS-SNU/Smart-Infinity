# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import torch

from deepspeed.utils.logging import logger
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed import comm as dist

from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, print_object, \
    get_sized_buffers, get_sized_buffer
from deepspeed.runtime.swap_tensor.async_swapper import AsyncTensorSwapper
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper

DEBUG_MODE = False

SWAP_IN_PARAM_TIMER = 'swap_in_param'
SWAP_OUT_PARAM_TIMER = 'swap_out_param'
SWAP_IN_GRADIENT_TIMER = 'swap_in_gradient'


class PartitionedOptimizerSwapper(OptimizerSwapper):

    def __init__(self, swap_config, aio_config, base_folder, optimizer, largest_numel, device, dtype, timers, use_fpga, num_ssds ):
        super(PartitionedOptimizerSwapper, self).__init__(swap_config, aio_config, base_folder, optimizer,
                                                          largest_numel, device, dtype, timers, use_fpga, num_ssds)
        self.aio_config = aio_config
        aio_op = AsyncIOBuilder().load()
        self.aio_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH],
                                            aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS],
                                            aio_config[AIO_THREAD_COUNT])
        
        # Overlap swapping out
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)
        # Overlap swapping out
        self.gradient_idx_swapper = None

        self.print_exclude_list += ['aio_handle', 'gradient_swapper', 'gradient_idx_swapper' 'print_exclude_list']

        if dist.get_rank() == 0:
            print_object(obj=self, name='PartitionedOptimizerSwapper', exclude_list=self.print_exclude_list)
        
        self.allocated_swap_buffers = []

    def initialize_parameters(self, parameters, src_tensors):
        self._initialize_parameters(parameters=parameters, src_tensors=src_tensors, aio_handle=self.aio_handle)

    def initialize_from_swapped_fp16_params(self, fp16_partitions_info, fp16_num_elems, fp16_pinned_buffers,
                                            fp32_parameters):
        self._initialize_from_swapped_fp16_params(aio_handle=self.aio_handle,
                                                  fp16_partitions_info=fp16_partitions_info,
                                                  fp16_num_elems=fp16_num_elems,
                                                  fp16_pinned_buffers=fp16_pinned_buffers,
                                                  fp32_parameters=fp32_parameters)

    def flush_gradients(self, use_fpga =False):
        self._flush_gradient_swapper(self.gradient_swapper, use_fpga)
        if self.gradient_idx_swapper is not None:
            self._flush_gradient_swapper(self.gradient_idx_swapper, use_fpga)

    def swap_in_optimizer_state(self, parameter, async_parameter=None, use_fpga=False):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return
        
        #print( dist.get_rank(), "=> I'm there 1!")
        #print("[Flush is problem in ",dist.get_rank()," ]")
        self._flush_gradient_swapper(self.gradient_swapper, use_fpga)
        if self.gradient_idx_swapper is not None:
            self._flush_gradient_swapper(self.gradient_idx_swapper, use_fpga)
        
        #print( dist.get_rank(), "=> I'm there 2!")
        assert(swap_info.has_gradients)
        
        required_buffer_count = len(swap_info.tensors)  + (1 if swap_info.has_gradients else 0 )
        
        #print( dist.get_rank(), "=> I'm there 3!")

        #print("[Allocate is problem in ",dist.get_rank()," ]")
        aligned_numel = self._io_aligned_numel(swap_info.numel())
        pinned_buffers = self.swap_buffer_manager.allocate(num_elems=aligned_numel,
                                                           count=required_buffer_count,
                                                           dtype=parameter.dtype)
        #print( dist.get_rank(), "=> I'm there 3!")
        assert pinned_buffers is not None
        #self.allocated_swap_buffers = pinned_buffers.copy()
        self.allocated_swap_buffers += pinned_buffers
        
        if use_fpga == False:
            self._start_timer(SWAP_IN_PARAM_TIMER)
        
        #print( dist.get_rank(), "=> I'm there 4!")

        #print("[swap_in_parameter is problem in ", dist.get_rank() ," ]")
        self._swap_in_parameter(aio_handle=self.aio_handle,
                            parameter=parameter,
                            dest_buffers=pinned_buffers[:required_buffer_count],
                            use_fpga = use_fpga)
        
        #print( dist.get_rank(), "=> I'm there 5!")

        if use_fpga == False:
            self._stop_timer(SWAP_IN_PARAM_TIMER)
            self.timer_names.add(SWAP_IN_PARAM_TIMER)

            self._start_timer(SWAP_IN_GRADIENT_TIMER)

        #print( dist.get_rank(), "=> I'm there 6!")
        #print("[swap_in_pinned_gradients is problem in ",dist.get_rank()," ]")
        self._swap_in_pinned_gradients(aio_handle=self.aio_handle, parameter=parameter, gradient_tensor= pinned_buffers[-1], use_fpga = use_fpga)
        #print( dist.get_rank(), "=> I'm there 7!")
        
        if use_fpga == False:
            self._stop_timer(SWAP_IN_GRADIENT_TIMER)
            self.timer_names.add(SWAP_IN_GRADIENT_TIMER)

    def swap_out_optimizer_state(self, parameter, async_swap=False, use_fpga= False):
        swap_info = self._get_param_swap_info(parameter=parameter)

        if swap_info is None:
            return
        
        if not use_fpga :
            self._start_timer(SWAP_OUT_PARAM_TIMER)

            pinned_tensors, pinned_paths, unpinned_tensors, unpinned_paths = self._separate_pinned_tensors(swap_info)
            swap_bytes = sum([self._io_aligned_numel(t.numel()) * t.element_size() for t in swap_info.tensors])

            WRITE_TIMER = 'swap_submit_write'
            self._start_timer(WRITE_TIMER)

            #print("[swap_out_tensor is problem in ", dist.get_rank() ," ]")
            swap_out_tensors(self.aio_handle, pinned_tensors, pinned_paths)
            #print("[aio_handle wait is problem in ", dist.get_rank() ," ]")
            assert self.aio_handle.wait() == len(pinned_tensors)
            for t in pinned_tensors:
                t.data = torch.Tensor()

            #print("[Allocate is problem in ", dist.get_rank() ," ]")
            if len(unpinned_tensors) > 0:
                pinned_buffers = self.swap_buffer_manager.allocate_all(num_elems=self.largest_numel, dtype=self.dtype)
                self._swap_out_unpinned_tensors(aio_handle=self.aio_handle,
                                                unpinned_tensors=unpinned_tensors,
                                                dest_paths=unpinned_paths,
                                                pinned_buffers=pinned_buffers)
                self.allocated_swap_buffers += pinned_buffers

                for t in unpinned_tensors:
                    t.data = torch.Tensor()
            self._stop_timer(WRITE_TIMER)
            self._log_timers([WRITE_TIMER])

        self.swap_buffer_manager.free(self.allocated_swap_buffers)
        self.allocated_swap_buffers = []
        

        if not use_fpga :
            self._stop_timer(SWAP_OUT_PARAM_TIMER)
            self.timer_names.add(SWAP_OUT_PARAM_TIMER)

        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_out: {(swap_bytes/(1024**3)):5.2f} GB')

    def swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors, use_fpga=False, comp_ratio = 2. ):
        if comp_ratio < 0.5 and self.gradient_idx_swapper is None:
            idx_aio_op = AsyncIOBuilder().load()
            self.idx_aio_handle = idx_aio_op.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                            self.aio_config[AIO_SINGLE_SUBMIT], self.aio_config[AIO_OVERLAP_EVENTS],
                                            self.aio_config[AIO_THREAD_COUNT])

            self.gradient_idx_swapper = AsyncTensorSwapper(aio_handle=self.idx_aio_handle,
            #self.gradient_idx_swapper = AsyncTensorSwapper(aio_handle=self.aio_handle,
                                   numel_alignment=self.numel_alignment,
                                   timers=self.timers)

        self._swap_out_gradients(parameter=parameter,
                                 gradient_offsets=gradient_offsets,
                                 gradient_tensors=gradient_tensors,
                                 gradient_swapper=self.gradient_swapper,
                                 gradient_idx_swapper = self.gradient_idx_swapper,
                                 use_fpga=use_fpga, 
                                 comp_ratio = comp_ratio
                                 )

    def _swap_in_parameter(self, aio_handle, parameter, dest_buffers, use_fpga =False):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return

        assert len(swap_info.tensors) <= len(dest_buffers)
    
        swap_lengths = [self._io_aligned_numel(swap_info.numel())] * len(swap_info.tensors)
        swap_buffers = get_sized_buffers(dest_buffers, swap_lengths)

        READ_TIMER = 'swap_submit_read_param'
        WAIT_TIMER = 'swap_wait_read_param'

        if not use_fpga:
            self._start_timer(READ_TIMER)
            swap_in_tensors(aio_handle, swap_buffers, swap_info.swap_paths)
            self._stop_timer(READ_TIMER)

        swap_bytes = sum([buffer.numel() * buffer.element_size() for buffer in swap_buffers])

        if not use_fpga:
            self._start_timer(WAIT_TIMER)
            #print("Start")
            #print("[aio is problem in ", dist.get_rank() ," ]")
            #aio_handle.wait()
            assert len(swap_buffers) == aio_handle.wait()
            #print("End")
            self._stop_timer(WAIT_TIMER)

        compute_lengths = [swap_info.numel()] * len(swap_info.tensors)

        compute_buffers = get_sized_buffers(dest_buffers, compute_lengths)
        for t, buffer in zip(swap_info.tensors, compute_buffers):
            t.data = buffer.data

        self._log_timers([READ_TIMER, WAIT_TIMER])
        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_in: {(swap_bytes/(1024**3)):5.2f} GB')

    def _separate_pinned_tensors(self, swap_info):
        pinned_tensors = []
        pinned_paths = []

        unpinned_tensors = []
        unpinned_paths = []

        for tensor, path in zip(swap_info.tensors, swap_info.swap_paths):
            if tensor.is_pinned():
                pinned_tensors.append(tensor)
                pinned_paths.append(path)
            else:
                unpinned_tensors.append(tensor)
                unpinned_paths.append(path)

        return pinned_tensors, pinned_paths, unpinned_tensors, unpinned_paths

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)


    def _swap_in_pinned_gradients(self, aio_handle, parameter, gradient_tensor, use_fpga=False):
        swap_info = self.swap_params_info[id(parameter)]
        if not (swap_info and swap_info.has_gradients()):
            return

        if use_fpga == False:
            if swap_info.swapped_gradients:
                param_gradients = swap_info.swapped_gradients.values()
                swap_buffers = [gradient_tensor.narrow(0, grad.offset, grad.length) for grad in param_gradients]
                swap_paths = [grad.path for grad in param_gradients]
                SWAP_READ_GRADIENTS = 'swap_submit_read_gradient'
                SWAP_WAIT_GRADIENTS = 'swap_submit_wait_gradient'

                self._start_timer(SWAP_READ_GRADIENTS)
                swap_in_tensors(aio_handle, swap_buffers, swap_paths)
                self._stop_timer(SWAP_READ_GRADIENTS)
                self._start_timer(SWAP_WAIT_GRADIENTS)
                assert len(swap_buffers) == aio_handle.wait()
                self._stop_timer(SWAP_WAIT_GRADIENTS)

                self._log_timers([SWAP_READ_GRADIENTS, SWAP_WAIT_GRADIENTS])

                parameter.grad = gradient_tensor.narrow(0, 0, parameter.numel()) 

    def _swap_in_gradients(self, aio_handle, parameter, dest_buffers):
        swap_info = self.swap_params_info.get(id(parameter), None)
        if not (swap_info and swap_info.has_gradients()):
            return
        
        assert(swap_info.swapped_gradients)

        param_gradients = swap_info.swapped_gradients.values()

        parameter.grad = dest_buffers[1].narrow(0, 0, parameter.numel()) 

        SWAP_READ_GRADIENTS = 'swap_submit_read_gradient'
        SWAP_WAIT_GRADIENTS = 'swap_submit_wait_gradient'

        for grad in param_gradients:
            swap_length = self._io_aligned_numel(grad.length)
            gradient_buffer = get_sized_buffer(dest_buffers[0], swap_length)
            
            swap_buffer = gradient_buffer.narrow(0, 0, swap_length)
            swap_path = grad.path

            self._start_timer(SWAP_READ_GRADIENTS)
            swap_in_tensors(aio_handle, [swap_buffer], [swap_path])
            self._stop_timer(SWAP_READ_GRADIENTS)

            self._start_timer(SWAP_WAIT_GRADIENTS)
            assert len([swap_buffer]) == aio_handle.wait()
            self._stop_timer(SWAP_WAIT_GRADIENTS)

            parameter.grad.narrow(0, grad.offset, grad.length).data = swap_buffer[:grad.length].data
        
        self._log_timers([SWAP_READ_GRADIENTS, SWAP_WAIT_GRADIENTS])

        #####################################
        #parameter.grad = dest_buffer.narrow(0, 0, parameter.numel())
        #if swap_info.unswapped_gradients:
        #    self._retrieve_unswapped_grad_partitions(swap_info=swap_info, dest_buffer=parameter.grad)
