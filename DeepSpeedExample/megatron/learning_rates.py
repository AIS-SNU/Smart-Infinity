# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learning rate decay functions."""

import math

from megatron import print_rank_0, get_args


class AnnealingLR(object):
    """Anneals the learning rate."""

    def __init__(self, optimizer, start_lr,
                 warmup_iter, total_iters,
                 decay_style, last_iter, min_lr=0.0,
                 use_checkpoint_lr_scheduler=True,
                 override_lr_scheduler=False):
        args = get_args()
        # Class values.
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.lr_decay_tokens = args.lr_decay_tokens
        self.num_tokens = 0
        self.warmup_tokens = 0
        self.decay_style = decay_style
        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, 'both override and '\
                'use-checkpoint are set.'
        # Set the learning rate
        self.step(self.num_iters, self.num_tokens)

        print_rank_0('> learning rate decay style: {}'.format(self.decay_style))

    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Warmup.
        if self.warmup_iter > 0:
            if self.num_iters == self.warmup_iter and self.lr_decay_tokens is not None:
                self.warmup_tokens = self.num_tokens
            if self.num_iters <= self.warmup_iter:
                return float(self.start_lr) * self.num_iters / self.warmup_iter

        if self.lr_decay_tokens is None:
            # For any iterations larger than `self.end_iter`, use `self.min_lr`.
            if self.num_iters > self.end_iter:
                return self.min_lr
            # If we are done with the warmup period, use the decay style.
            current_iter = self.num_iters - self.warmup_iter
            decay_iter = self.end_iter - self.warmup_iter
            decay_ratio = float(current_iter) / float(decay_iter)
        else:
            if self.num_tokens > self.lr_decay_tokens:
                return self.min_lr
            current_tokens = self.num_tokens - self.warmup_tokens
            decay_tokens = self.lr_decay_tokens - self.warmup_tokens
            decay_ratio = float(current_tokens) / float(decay_tokens)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0

        if self.decay_style == 'linear':
            lr = self.start_lr * (1.0 - decay_ratio)
        elif self.decay_style == 'cosine':
            lr = self.start_lr / 2.0 * (math.cos(
                math.pi * decay_ratio) + 1)
        elif self.decay_style == 'exponential':
            # exp(-0.693) = 1/2
            lr = self.start_lr * math.exp(-0.693 * decay_ratio)
        else:
            lr = self.start_lr
        return max(lr, self.min_lr)

    def step(self, step_num=None, token_num=None):
        """Set lr for all parameters groups."""
        args = get_args()
        if step_num is None:
            step_num = self.num_iters + 1
        if token_num is None:
            token_num = args.tokens
        self.num_iters = step_num
        self.num_tokens = token_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        state_dict = {
            'start_lr': self.start_lr,
            'warmup_iter': self.warmup_iter,
            'num_iters': self.num_iters,
            'warmup_tokens': self.warmup_tokens,
            'num_tokens': self.num_tokens,
            'decay_style': self.decay_style,
            'end_iter': self.end_iter,
            'min_lr': self.min_lr
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, 'AnnealingLR: class input value' \
                'and checkpoint values for {} do not match'.format(name)
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value

    def load_state_dict(self, sd):

        self.start_lr = self._check_and_set(self.start_lr, sd['start_lr'],
                                            'learning rate')
        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')
        self.warmup_iter = self._check_and_set(self.warmup_iter,
                                               sd['warmup_iter'],
                                               'warmup iterations')
        self.end_iter = self._check_and_set(self.end_iter, sd['end_iter'],
                                            'total number of iterations')
        self.decay_style = self._check_and_set(self.decay_style,
                                               sd['decay_style'],
                                               'decay style')

        self.num_iters = sd['num_iters']
        if 'warmup_tokens' in sd:
            self.warmup_tokens = sd['warmup_tokens']
        if 'num_tokens' in sd:
            self.num_tokens = sd['num_tokens']
        self.step(self.num_iters, self.num_tokens)
