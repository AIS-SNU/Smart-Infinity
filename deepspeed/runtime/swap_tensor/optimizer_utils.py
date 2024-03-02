# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import torch

from deepspeed import comm as dist
from deepspeed.utils.logging import logger
from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, \
    MIN_AIO_BYTES, AIO_ALIGNED_BYTES, get_sized_buffers
from deepspeed.runtime.swap_tensor.utils import SwapBufferManager, SwapBufferPool

from pathlib import Path

class FlattenedTensorSwapInfo(object):

    def __init__(self, path, length, offset):
        self.path = path
        self.offset = offset
        self.length = length


class OptimizerStateSwapInfo(object):

    def __init__(self, parameter, numel, base_folder, sub_group_id = 0, use_fpga = False, num_ssds = 1):
        self.tensors = []
        self.param_id = id(parameter)
        self.use_fpga = use_fpga 
        self.num_ssds = num_ssds
        
        #base_folder = os.path.join(base_folder, 'zero_stage_3')

        if self.use_fpga == 1:
            nvme_swap_folder = os.path.join(base_folder, 'smartssd')
            self.swap_folder = Path(str(nvme_swap_folder) + str(sub_group_id % self.num_ssds))
        elif self.use_fpga == 0:
            nvme_swap_folder = os.path.join(base_folder, 'raid')
            self.swap_folder = Path(str(nvme_swap_folder) + str(self.num_ssds))
        else:
            raise NotImplementedError

        self.swap_paths = []
        self.swapped_gradients = {}
        self.unswapped_gradients = {}
        self.tensor_numel = numel
        
        self.tensor_dtype = parameter.dtype

        self.tensor_device = parameter.device
        self.has_state_tensors = False
        self._add_tensors([parameter])

    def numel(self):
        return self.tensor_numel

    def has_gradients(self):
        return self.swapped_gradients or self.unswapped_gradients

    def _add_tensors(self, tensor_list, tensor_type = 'param'):
        assert(len(tensor_list) == 1)
        for t in tensor_list:
            self.tensors.append(t)
            d_path = os.path.join(self.swap_folder, str(dist.get_rank()) )
            os.makedirs(d_path, exist_ok=True)
            p_path = os.path.join(d_path, f'{self.param_id}')
            os.makedirs(p_path, exist_ok=True)
            path = os.path.join(p_path,f'{tensor_type}.tensor.swp')

            self.swap_paths.append(path)

    def add_state_tensors(self, tensor_dict):
        self.has_state_tensors = True
        for k, t in tensor_dict.items():
            self._add_tensors([t], tensor_type = k)

    def device(self):
        return self.tensor_device

    def dtype(self):
        return self.tensor_dtype

    def release_memory(self):
        for tensor in self.tensors:
            tensor.data = torch.Tensor()

    def get_or_create_gradient_paths(self, offsets, lengths):
        gradient_paths = []
        for offset, length in zip(offsets, lengths):
            if not offset in self.swapped_gradients.keys():
                d_path = os.path.join(self.swap_folder, str(dist.get_rank()))
                os.makedirs(d_path, exist_ok=True)
                p_path = os.path.join(d_path, f'{self.param_id}' )
                os.makedirs(p_path, exist_ok=True)

                path = os.path.join( p_path, f'gradient_tensor.swp{offset}' )
                self.swapped_gradients[offset] = FlattenedTensorSwapInfo(path, length, offset)

            gradient_paths.append(self.swapped_gradients[offset].path)

        return gradient_paths

    def set_swap_buffers(self, buffers):
        compute_lengths = [self.numel()] * len(self.tensors)
        compute_buffers = get_sized_buffers(buffers, compute_lengths)
        for t, buffer in zip(self.tensors, compute_buffers):
            t.data = buffer.data

    def get_swap_gradient_buffers(self, swap_buffer):
        assert self.numel() <= swap_buffer.numel()
        return [swap_buffer.narrow(0, grad.offset, grad.length) for grad in self.swapped_gradients.values()]

    def get_swap_gradient_paths(self):
        return [grad.path for grad in self.swapped_gradients.values()]

    def get_unpinned_state_tensors(self):
        return [t for t in self.tensors if not t.is_pinned()]

    def read_unswapped_gradients(self, dest_buffer):
        num_elem_count = 0
        for offset, grad_partition in self.unswapped_gradients.items():
            dst_tensor = dest_buffer.narrow(0, offset, grad_partition.numel())
            dst_tensor.data.copy_(grad_partition.data)
            num_elem_count += grad_partition.numel()

        return num_elem_count

    def release_unswapped_gradients(self):
        self.unswapped_gradients = {}


SWAPPER_DEBUG_MODE = False
SWAP_OUT_GRADIENT_TIMER = 'swap_out_gradient'


class OptimizerSwapper(object):

    def __init__(self, swap_config, aio_config, base_folder, optimizer, largest_numel, device, dtype, timers, use_fpga, num_ssds):
        self.swap_config = swap_config
        self.aio_config = aio_config
        
        # SmartInfinity 
        self.use_fpga = use_fpga
        self.num_ssds = num_ssds

        # NVMe swap management
        self.swap_params_info = {}
        self.swap_element_size = torch.tensor([], dtype=dtype).element_size()
        
        self.swap_folder = base_folder
        #os.makedirs(self.swap_folder, exist_ok=True)

        self.optimizer = optimizer

        # Read/Write alignment for each thread during Intra-request parallelism
        self.min_aio_bytes = max(MIN_AIO_BYTES, aio_config[AIO_BLOCK_SIZE])
        self.aligned_bytes = AIO_ALIGNED_BYTES * aio_config[AIO_THREAD_COUNT]
        self.numel_alignment =( self.aligned_bytes // self.swap_element_size ) 
        assert(self.numel_alignment == 1024)

        # Swap buffer management
        self.largest_numel = self._io_aligned_numel(largest_numel)

        self.dtype = dtype

        self.swap_buffer_manager = SwapBufferManager(num_elems=self.largest_numel,
                                                     count=swap_config.buffer_count,
                                                     dtype=dtype)
        if use_fpga: 
            self.swap_gradient_manager = None
            self.swap_idx_manager = None
        else:
            self.swap_gradient_manager = None

        # Timers
        self.timers = timers
        self.timer_names = set()

        # Print exclusion list
        self.print_exclude_list = [
            'optimizer',
            'swap_buffer_manager',
            'swap_params_info',
            'timers',
            'timer_names',
        ]

    def swappable_tensor(self, param=None, numel=None):
        assert param is not None or numel is not None, "Either param or numel must be provided"
        
        return True
        
        if param is not None:
            return self.min_aio_bytes <= (param.numel() * self.swap_element_size)
        return self.min_aio_bytes <= (numel * self.swap_element_size)

    def init_timers(self):
        self.timer_names = set()

    def log_timers(self):
        if self.timer_names:
            self._log_timers(list(self.timer_names), force=True)

    def pre_backward(self):
        self.init_timers()

    def post_backward(self):
        pass

    def _flush_gradient_swapper(self, gradient_swapper, use_fpga=False):
        if gradient_swapper.has_buffers():
            if not use_fpga:
                self._start_timer(SWAP_OUT_GRADIENT_TIMER)
            pinned_buffers = gradient_swapper.release_buffers()
            if use_fpga:
                if self.swap_idx_manager is not None:
                    for tensor in pinned_buffers:
                        if tensor.dtype == torch.int32:
                            self.swap_idx_manager.free( [tensor] )
                        elif tensor.dtype == torch.float16:
                            self.swap_gradient_manager.free( [tensor] )
                        else:
                            raise NotImplementedError

                else:
                    self.swap_gradient_manager.free(pinned_buffers)
            else:             
                self.swap_buffer_manager.free(pinned_buffers)
                self._stop_timer(SWAP_OUT_GRADIENT_TIMER)
                self.timer_names.add(SWAP_OUT_GRADIENT_TIMER)
                self.timer_names.update(gradient_swapper.get_timer_names())

    def _swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors, gradient_swapper, gradient_idx_swapper, use_fpga=False, comp_ratio = 1. ):
        if not id(parameter) in self.swap_params_info.keys():
            return

        swap_info = self.swap_params_info[id(parameter)]
        for offset, tensor in zip(gradient_offsets, gradient_tensors):
            swap_info.unswapped_gradients[offset] = tensor
    
        if (gradient_offsets[0] > 0):
            return
        else:
            max_top_size = ( int (comp_ratio * self.largest_numel ) )
            max_aligned_top_size = self._io_aligned_numel( max_top_size )
            
            src_tensors  = sorted(swap_info.unswapped_gradients.items())

            #assert(swap_info.numel() == sum([v.numel() for k, v in src_tensors]))
            aligned_numel = self._io_aligned_numel(swap_info.numel())
            
            if use_fpga:
                if self.swap_gradient_manager is None:
                    if comp_ratio > 0.5:
                        self.swap_gradient_manager = SwapBufferManager(num_elems=self.largest_numel,
                                                     count=self.swap_config.buffer_count,
                                                     dtype=torch.float16)
                    else:
                        self.swap_gradient_manager = SwapBufferManager(num_elems=self.largest_numel,
                                                     count=self.swap_config.buffer_count,
                                                     dtype=torch.float16)



                if comp_ratio > 0.5:
                    gradient_tensors = self.swap_gradient_manager.allocate(num_elems=aligned_numel,
                                                           count=1,
                                                            dtype=torch.float16)
                else:
                    gradient_tensors = self.swap_gradient_manager.allocate(num_elems= aligned_numel,
                                                           count=1,
                                                            dtype=torch.float16)

                
            else:
                gradient_tensors = self.swap_buffer_manager.allocate(num_elems=aligned_numel,
                                                           count=1,
                                                            dtype=self.dtype)

            assert gradient_tensors is not None
            for src_key, src_val in src_tensors:
                gradient_tensors[0].narrow(0, src_key, len(src_val)).copy_(src_val)
            
            if comp_ratio < 0.5: 
                if use_fpga:
                    if self.swap_idx_manager is None:
                        self.swap_idx_manager = SwapBufferManager(num_elems=max_aligned_top_size,
                                                     count=self.swap_config.buffer_count,
                                                     dtype=torch.int32)
                        
                    
                    # Do compression, offset= 0: value, offset1: idx
                    contiguous_grad = gradient_tensors[0].to('cuda')
                    top_size = ( int (comp_ratio * contiguous_grad.numel()) )
                    aligned_top_size = self._io_aligned_numel( top_size )
                    
                    #_, positions = torch.topk(contiguous_grad.abs(), aligned_top_size, sorted= True )
                    #_, positions = torch.topk(contiguous_grad.abs(), aligned_top_size, sorted= True )
                    _, idx = contiguous_grad.abs().sort(descending=True)

                    positions = idx[:aligned_top_size] 

                    values = contiguous_grad[positions].to('cpu')
                    ipositions = positions.to(torch.int32).to('cpu')

                    #swap_info.ipositions = ipositions 

                    self.swap_gradient_manager.free(gradient_tensors)
                    gradient_tensors = []
                    
                    gradient_tensors = self.swap_idx_manager.allocate(num_elems=aligned_top_size,
                                                           count=1,
                                                            dtype=torch.int32)

                    gradient_tensors_fp16 = self.swap_gradient_manager.allocate(num_elems=aligned_top_size,
                                                           count=1,
                                                            dtype=torch.float16)
                    
                    
                    gradient_tensors.append( gradient_tensors_fp16[0] )
                    
                    gradient_offsets = [ 0, 1 ]
                    gradient_tensors[0].copy_(ipositions)
                    gradient_tensors[1].copy_(values)

                else:
                    # This is for sanity check for Topk compression ( Not Used )
                    contiguous_grad = gradient_tensors[0].to('cuda')
                    top_size = ( int (comp_ratio * contiguous_grad.numel()) )
                    aligned_top_size = self._io_aligned_numel( top_size )

                    _, idx = contiguous_grad.abs().sort(descending=True)
                    positions = idx[:aligned_top_size] 
                    #_, positions = torch.topk(contiguous_grad.view(-1).abs(), aligned_top_size, sorted= True)
                    
                    values = contiguous_grad[positions]
                    contiguous_grad.zero_();
                    contiguous_grad[positions] = values
                    contiguous_grad = contiguous_grad.to('cpu')
                    values = values.to('cpu')
                    positions = positions.to('cpu')  

                    gradient_offsets = [ 0 ]
                    gradient_tensors[0].copy_(contiguous_grad)

            else:
                # No compression
                gradient_offsets = [ 0 ]

            self.allocated_swap_buffers = gradient_tensors.copy()
            swap_info.unswapped_gradients = {}

        swappable_tensors = []
        swappable_offsets = []
        swappable_lengths = []

        aligned_gradients, aligned_offsets = self._adjust_for_misaligned_lengths(tensors=gradient_tensors,
                                                                                 offsets=gradient_offsets)
        self._start_timer(SWAP_OUT_GRADIENT_TIMER)
        for tensor, offset in zip(aligned_gradients, aligned_offsets):
            if not self.swappable_tensor(param=tensor):
                raise NotImplementedError 
                swap_info.unswapped_gradients[offset] = tensor
                continue

            swappable_tensors.append(tensor)
            swappable_offsets.append(offset)
            swappable_lengths.append(tensor.numel())
        
        if len(swappable_tensors) > 0:

            swappable_paths = swap_info.get_or_create_gradient_paths(swappable_offsets, swappable_lengths)

            if use_fpga and comp_ratio < 0.5:
                
                # Swap out idx in int32
                assert (gradient_idx_swapper is not None)

                if not gradient_idx_swapper.has_buffers():
                    pinned_buffers = self.swap_idx_manager.allocate_all(num_elems=max_aligned_top_size, dtype=torch.int32)
                    gradient_idx_swapper.add_buffers(pinned_buffers)
                
                gradient_idx_swapper.swap_out_tensors(tensor_list=swappable_tensors[:1], path_list=swappable_paths[:1])

                swappable_paths = swappable_paths[1:] # Slice out index path
                swappable_tensors = swappable_tensors[1:] # Slice out index tensor
                swappable_lengths = swappable_lengths[1:] # Slice out index tensor
                

            if not gradient_swapper.has_buffers():
                if use_fpga:
                    if comp_ratio < 0.5:
                        pinned_buffers = self.swap_gradient_manager.allocate_all(num_elems=max_aligned_top_size, dtype=torch.float16)
                    else:
                        pinned_buffers = self.swap_gradient_manager.allocate_all(num_elems= self._io_aligned_numel(self.largest_numel), dtype=torch.float16)
                else:
                    pinned_buffers = self.swap_buffer_manager.allocate_all(num_elems=self._io_aligned_numel(self.largest_numel), dtype=self.dtype)

                gradient_swapper.add_buffers(pinned_buffers)

            gradient_swapper.swap_out_tensors(tensor_list=swappable_tensors, path_list=swappable_paths)

        if use_fpga:
            if comp_ratio < 0.5: 
                self.swap_idx_manager.free([self.allocated_swap_buffers[0]])
                self.swap_gradient_manager.free([self.allocated_swap_buffers[1]])
            else:
                self.swap_gradient_manager.free(self.allocated_swap_buffers)
        else:
            self.swap_buffer_manager.free(self.allocated_swap_buffers)
        self.allocated_swap_buffers = []
            
        self._stop_timer(SWAP_OUT_GRADIENT_TIMER)
        self.timer_names.add(SWAP_OUT_GRADIENT_TIMER)

    def _initialize_from_swapped_fp16_params(self, aio_handle, fp16_partitions_info, fp16_num_elems,
                                             fp16_pinned_buffers, fp32_parameters):
        assert len(fp32_parameters) == len(fp16_partitions_info)
        assert len(fp32_parameters) == len(fp16_num_elems)
        assert all([buffer.is_pinned() for buffer in fp16_pinned_buffers])

        fp32_swap_paths = self._get_swap_paths(parameters=fp32_parameters, num_elems=fp16_num_elems)

        fp32_pinned_buffers = self.swap_buffer_manager.allocate_all(num_elems=self.largest_numel, dtype=self.dtype)

        fp16_buffer_numel = [buf.numel() for buf in fp16_pinned_buffers]
        assert all([numel >= self.largest_numel for numel in fp16_buffer_numel]), \
        f"numel of fp16 buffers {fp16_buffer_numel} is too small for initializing fp32 params {self.largest_numel}"

        fp32_swap_buffers = SwapBufferPool(fp32_pinned_buffers)
        fp16_swap_buffers = SwapBufferPool(fp16_pinned_buffers)

        curr_index = 0
        while curr_index < len(fp32_parameters):
            fp16_pinned_tensors = self._swap_in_fp16_params(aio_handle=aio_handle,
                                                            fp16_num_elems=fp16_num_elems[curr_index:],
                                                            fp16_partitions_info=fp16_partitions_info[curr_index:],
                                                            fp16_swap_buffers=fp16_swap_buffers)

            if dist.get_rank() == 0 and SWAPPER_DEBUG_MODE:
                for i, tensor in enumerate(fp16_pinned_tensors):
                    true_index = curr_index + i
                    logger.info(
                        f'swap_in_fp16_param: fp32_id = {id(fp32_parameters[true_index])} index = {true_index} orig_num_elem = {fp16_num_elems[true_index]}, swap_num_elem = {fp16_pinned_tensors[i].numel()}'
                    )

            swap_out_count = self._swap_out_fp16_params(aio_handle=aio_handle,
                                                        fp32_swap_paths=fp32_swap_paths[curr_index:],
                                                        fp32_swap_buffers=fp32_swap_buffers,
                                                        fp16_pinned_tensors=fp16_pinned_tensors)
            assert swap_out_count == len(fp16_pinned_tensors), \
            f"{swap_out_count} does not match {len(fp16_pinned_tensors)}"

            fp16_swap_buffers.reset()
            fp32_swap_buffers.reset()
            curr_index += swap_out_count

        self.swap_buffer_manager.free(fp32_pinned_buffers)

    def _swap_in_fp16_params(self, aio_handle, fp16_num_elems, fp16_partitions_info, fp16_swap_buffers):
        assert len(fp16_num_elems) > 0

        swapped_fp16_tensors = []
        swap_tensors = []
        swap_paths = []
        unswapped_srcs = []
        unswapped_dsts = []

        for i, numel in enumerate(fp16_num_elems):
            pinned_tensor, _ = fp16_swap_buffers.allocate_tensor(numel, None, numel)
            if pinned_tensor is None:
                break

            swapped_fp16_tensors.append(pinned_tensor)
            offset = 0
            for tensor, partition_numel, partition_path in fp16_partitions_info[i]:
                dst_tensor = pinned_tensor.narrow(0, offset, partition_numel)
                if partition_path is None:
                    unswapped_srcs.append(tensor)
                    unswapped_dsts.append(dst_tensor)
                else:
                    swap_paths.append(partition_path)
                    swap_tensors.append(dst_tensor)
                offset += partition_numel

        assert len(swapped_fp16_tensors) + len(unswapped_srcs) > 0
        ret = swap_in_tensors(aio_handle, swap_tensors, swap_paths)
        for src, dst in zip(unswapped_srcs, unswapped_dsts):
            dst.data.copy_(src.data)

        assert len(swap_tensors) == aio_handle.wait()

        return swapped_fp16_tensors

    def _swap_out_fp16_params(self, aio_handle, fp32_swap_paths, fp32_swap_buffers, fp16_pinned_tensors):

        assert len(fp16_pinned_tensors) <= len(fp32_swap_paths)
        swap_out_count = 0
        for i, fp16_tensor in enumerate(fp16_pinned_tensors):
            if not fp32_swap_buffers.has_space(fp16_tensor.numel()):
                fp32_swap_buffers.swap_out(aio_handle)
                fp32_swap_buffers.reset()

            pinned_tensor, _ = fp32_swap_buffers.insert_tensor(fp16_tensor, fp32_swap_paths[i],
                                                               self._io_aligned_numel(fp16_tensor.numel()))
            assert pinned_tensor is not None
            swap_out_count += 1

        if len(fp32_swap_buffers.get_swap_tensors()) > 0:
            fp32_swap_buffers.swap_out(aio_handle)

        return swap_out_count

    def _initialize_parameters(self, parameters, src_tensors, aio_handle):
        assert len(parameters) == len(src_tensors)

        swap_paths = self._get_swap_paths(parameters=parameters, num_elems=[src.numel() for src in src_tensors])

        SWAP_INIT_TIMER = "swap_init_write"
        self._start_timer(SWAP_INIT_TIMER)

        pinned_buffers = self.swap_buffer_manager.allocate_all(num_elems=self.largest_numel, dtype=self.dtype)
        assert pinned_buffers is not None

        self._swap_out_unpinned_tensors(aio_handle=aio_handle,
                                        unpinned_tensors=src_tensors,
                                        dest_paths=swap_paths,
                                        pinned_buffers=pinned_buffers)

        if dist.get_rank() == 0 and SWAPPER_DEBUG_MODE:
            for i, tensor in enumerate(src_tensors):
                logger.info(
                    f'copy_in_fp16_param: fp32_id = {id(parameters[i])} index = {i}, swap_num_elem = {src_tensors[i].numel()}'
                )

        self.swap_buffer_manager.free(pinned_buffers)

        self._stop_timer(SWAP_INIT_TIMER)
        self._log_timers([SWAP_INIT_TIMER])

    def _get_swap_paths(self, parameters, num_elems):
        swap_info_list = [
            self._create_param_swap_info(parameter=p,
                                         numel=numel,
                                         sub_group_id = sub_group_id) \
            for sub_group_id, (p, numel) in enumerate(zip(parameters, num_elems))
        ]
        assert len(swap_info_list) == len(num_elems)

        swap_paths = [info.swap_paths[0] for info in swap_info_list]
        return swap_paths

    def _swap_out_unpinned_tensors(self, aio_handle, unpinned_tensors, dest_paths, pinned_buffers):

        swap_buffer_count = len(pinned_buffers)
        unpinned_tensor_count = len(unpinned_tensors)

        for i in range(0, unpinned_tensor_count, swap_buffer_count):
            swap_tensor_count = min((unpinned_tensor_count - i), swap_buffer_count)

            src_tensors = unpinned_tensors[i:(i + swap_tensor_count)]
            compute_lengths = [t.numel() for t in src_tensors]
            compute_buffers = get_sized_buffers(pinned_buffers, compute_lengths)

            for dst, src in zip(compute_buffers, src_tensors):
                dst.data.copy_(src.data)

            swap_lengths = [self._io_aligned_numel(t.numel()) for t in src_tensors]
            swap_buffers = get_sized_buffers(pinned_buffers, swap_lengths)

            swap_paths = dest_paths[i:(i + swap_tensor_count)]
            swap_out_tensors(aio_handle, swap_buffers, swap_paths)

            assert aio_handle.wait() == swap_tensor_count

    def _adjust_for_misaligned_lengths(self, tensors, offsets):
        new_tensors = []
        new_offsets = []

        for orig_tensor, orig_offset in zip(tensors, offsets):
            if not self.swappable_tensor(param=orig_tensor):
                raise NotImplementedError
                new_tensors.append(orig_tensor)
                new_offsets.append(orig_offset)
                continue

            remainder = orig_tensor.numel() % self.numel_alignment
            if remainder == 0:
                new_tensors.append(orig_tensor)
                new_offsets.append(orig_offset)
                continue

            raise NotImplementedError
            # Split into two by making remainder a tensor
            aligned_length = (orig_tensor.numel() // self.numel_alignment) * self.numel_alignment
            new_tensors.append(orig_tensor.narrow(0, 0, aligned_length))
            new_offsets.append(orig_offset)

            # remainder tensor
            new_tensors.append(orig_tensor.narrow(0, aligned_length, remainder))
            new_offsets.append(orig_offset + aligned_length)

        return new_tensors, new_offsets

    def _retrieve_unswapped_grad_partitions(self, swap_info, dest_buffer):
        UNSWAPPED_READ_GRADIENTS = 'unswapped_read_gradients'
        self._start_timer(UNSWAPPED_READ_GRADIENTS)
        tensor_count = len(swap_info.unswapped_gradients)
        num_elem_count = swap_info.read_unswapped_gradients(dest_buffer)
        self._stop_timer(UNSWAPPED_READ_GRADIENTS)
        self._log_timers([UNSWAPPED_READ_GRADIENTS])

        # It should be safe to discard unswapped gradient partitions
        swap_info.release_unswapped_gradients()

        if SWAPPER_DEBUG_MODE:
            logger.info(
                f'optimizer_retrieve_unswapped_gradients: param={swap_info.param_id} tensor_count={tensor_count} elem_count={num_elem_count}'
            )

    def _get_state_tensors(self, parameter):
        if not parameter in self.optimizer.state:
            return {}

        tensor_dict = {}
        for key, value in self.optimizer.state[parameter].items():
            if torch.is_tensor(value):
                tensor_dict[key] = value

        return tensor_dict

    def _update_param_state_info(self, swap_info, parameter):
        if not swap_info.has_state_tensors:
            state_tensors = self._get_state_tensors(parameter)
            if state_tensors:
                swap_info.add_state_tensors(state_tensors)

    def _create_param_swap_info(self, parameter, numel, sub_group_id = 0):
        param_id = id(parameter)
        assert not param_id in self.swap_params_info

        self.swap_params_info[param_id] = OptimizerStateSwapInfo(parameter=parameter,
                                                                 numel=numel,
                                                                 base_folder=self.swap_folder, 
                                                                 sub_group_id = sub_group_id,
                                                                 use_fpga = self.use_fpga,
                                                                 num_ssds = self.num_ssds)
        swap_info = self.swap_params_info[param_id]

        self._update_param_state_info(swap_info, parameter)

        return swap_info

    def _get_param_swap_info(self, parameter):
        param_id = id(parameter)
        swap_info = self.swap_params_info.get(param_id, None)

        if swap_info is not None:
            self._update_param_state_info(swap_info, parameter)

        return swap_info

    def _start_timer(self, name):
        if self.timers:
            self.timers(name).start()

    def _stop_timer(self, name):
        if self.timers:
            self.timers(name).stop()

    def _log_timers(self, name_list, force=False):
        if self.timers and (SWAPPER_DEBUG_MODE or force):
            self.timers.log(name_list)

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)
