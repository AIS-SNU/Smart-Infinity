#! /bin/bash

export DS_BUILD_AIO=1
export PATH=/usr/local/cuda/bin:$PATH


rm -rf ./checkpoints/* 

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_zero_stage_infinity-nvme.json"

# Megatron Model Parallelism
mp_size=1

NLAYERS=${1-24}
NHIDDEN=${2-2560}
HEADS=25
SEQ=${4-1024}
BATCHSIZE=${5-4}
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4"
_BASE=${7}
#_BASE=/home/sys/STRONGHOLD/data
jobname="bert-pile"
data_home="/mnt/data/bert_dataset"
#data_path="${data_home}/pile_bert_train_text_sentence"
data_path="/mnt/data/bert_dataset/my-bert_text_sentence"
vocab_path="bert-large-uncased-vocab.txt"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
fi
data_options=" \
    --vocab-file ${vocab_path} \
    --data-path ${data_path} \
	--data-impl mmap "

CHECKPOINT_PATH=checkpoints/gpt2_ds

#ZeRO Configs
stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false


gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads $HEADS \
        --seq-length $SEQ \
        --max-position-embeddings $SEQ \
        --batch-size $BATCHSIZE \
        --train-iters 50 \
        --log-interval 1 \
        --exit-interval 5 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 5.0e-7 \
        --lr-decay-style cosine \
        --min-lr 1.0e-8 \
        --weight-decay 1e-2 \
        --clip-grad 0.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --save-interval 10000 \
        --eval-interval 100 \
        --eval-iters 100 \
        --cpu-optimizer \
        --fp16  \
        "
        #--tensorboard-dir ${LOGDIR}
" 
"
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

export PYTHONGIL=1
run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_bert.py ${data_options} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
