#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1
LOCAL_RANK=0


_BASE=/home/sys/STRONGHOLD/data
DATA_PATH=${_BASE}/my-gpt2-en_text_document
VOCAB_PATH=${_BASE}/gpt2-vocab.json
MERGE_PATH=${_BASE}/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

NLAYERS=${1-24} 
NHIDDEN=${2-2560} 
HEADS=${3-16} 
SEQ=${4-1024} 
BATCHSIZE=${5-4} 
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4" 



python pretrain_gpt2.py \
       --num-layers ${NLAYERS} \
       --hidden-size ${NHIDDEN} \
       --num-attention-heads ${HEADS} \
       --batch-size ${BATCHSIZE} \
       --seq-length ${SEQ} \
       --max-position-embeddings ${SEQ} \
       --train-iters 50 \
       --log-interval 10 \
       --exit-interval 50 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ${VOCAB_PATH} \
       --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1000 

set +x
