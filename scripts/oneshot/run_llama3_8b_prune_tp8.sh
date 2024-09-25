#!/bin/bash

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45530" # select the port
NNODES=1 # number of nodes
NPROC_PER_NODE=8 # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SPARSEMETHOD=$1 # SparseGPT, Magnitude, Wanda
EXTRA_CMD=$2

export TASK='wikitext'
export SPARSITY=0.5
export PATTERN='nmprune'
export EXCLUDE=0

NSAMPLES=128
BASE_NAME="llama3-8b-tp8"
NAME="${BASE_NAME}.sparse.${PATTERN}.sp${SPARSITY}${SPARSEMETHOD}.ex${EXCLUDE}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
PROJECT_DIR=$(pwd)
LOG_DIR=$PROJECT_DIR/output/oneshot_pruning
mkdir -p $LOG_DIR

CHECKPOINT_LOAD_DIR="$PROJECT_DIR/assets/checkpoints/llama3_8b_megatron_tp8"
CHECKPOINT_SAVE_DIR="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${NAME}"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/llama3_8b_hf"

TASK_NAME="PRUNE-WIKITEXT2"
options=" \
    ${mag_options} \
    --task ${TASK_NAME} \
    --valid-data ${VALID_DATA} \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --use-rotary-position-embeddings \
    --swiglu \
    --rotary-base 500000 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --group-query-attention \
    --num-query-groups 8 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-iters 1000 \
    --log-interval 10 \
    --overlapping-eval 4096 \
    --eval-iters 10 \
    --eval-interval 500 \
    --tokenizer-type AutoTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --make-vocab-size-divisible-by 1 \
    --ffn-hidden-size 14336 --normalization RMSNorm \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --group-query-attention \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --data-path "None" \
    --save-interval 20000 \
    --save ${CHECKPOINT_SAVE_DIR} \
    --load ${CHECKPOINT_LOAD_DIR} \
    --hessian-compute \
    --sparse-pattern ${PATTERN} \
    --sparse-method ${SPARSEMETHOD} \
    --sparsity ${SPARSITY} \
    --row-b -1 \
    --col-b 128 \
    --prunen 2 \
    --prunem 4 \
    --hessian-samples $NSAMPLES \
    --exclude-layers-from-prune ${EXCLUDE} ${EXTRA_CMD} "

cd $PROJECT_DIR; export CUDA_DEVICE_MAX_CONNECTIONS=1; 

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tasks/main.py $options