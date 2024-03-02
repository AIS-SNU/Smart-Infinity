#!/bin/bash
# export PATH=/usr/local/cuda/bin:$PATH

# model configuration
NUM_LAYERS=20
HIDDEN_SIZE=2560
HEADS=16
SEQ_LEN=1024
BATCH_SIZE=4

DUMMY=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

_BASE=/mnt/data/Megatron-LM_data

while getopts 'm:l:h:b:a:s:n:f:c:o:' flag
do
    case "${flag}" in
        m) METHOD=${OPTARG};;
        l) NUM_LAYERS=${OPTARG};;
        h) HIDDEN_SIZE=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
		a) HEADS=${OPTARG};;
		s) SEQ_LEN=${OPTARG};;
		# SmartInfinity options
		n) N_SSD=${OPTARG};;
		f) USE_FPGA=${OPTARG};;
		c) COMP_RATIO=${OPTARG};;
		o) OPT_TYPE=${OPTARG};;
    esac
done

_LOG_DIR=${script_dir}/../results


if [[ 'zero-infinity-nvme' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../DeepSpeedExample/examples
    _SCRIPT=ds_pretrain_gpt2-infinity-nvme.sh

elif [[ 'zero-infinity-nvme-distributed' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../DeepSpeedExample/examples
    _SCRIPT=ds_pretrain_gpt2-infinity-nvme-distributed.sh

elif [[ 'zero-infinity-nvme-bert' = $METHOD ]]; then
    _SRC_DIR=${script_dir}/../DeepSpeedExample/examples
    _SCRIPT=ds_pretrain_bert-infinity-nvme.sh

else
    echo "the value of '-m' is illegal: $METHOD"
    exit 0
fi

CMD="cd ${_SRC_DIR}/.. && \
    ${_SRC_DIR}/${_SCRIPT} ${NUM_LAYERS} ${HIDDEN_SIZE} ${HEADS} ${SEQ_LEN} ${BATCH_SIZE} ${DUMMY} ${_BASE} \
		${N_SSD} ${USE_FPGA} ${COMP_RATIO} ${OPT_TYPE}\
		2>&1 | \
        tee ${_LOG_DIR}/log_${N_SSD}_${OPT_TYPE}_${USE_FPGA}_${COMP_RATIO}_l-${NUM_LAYERS}_hs-${HIDDEN_SIZE}_bs-${BATCH_SIZE}_m-${METHOD}_$(date '+%Y-%m-%d.%s').txt && \
    cd -"

echo $CMD
eval $CMD
