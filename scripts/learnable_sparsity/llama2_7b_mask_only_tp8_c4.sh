#!/usr/bin/bash

# Get Data Blend
. ./assets/c4-blend.sh # check this file for more detials about the training data
echo $DATA_BLEND 

export MASTER_ADDR="127.0.0.1" # select the master address
export MASTER_PORT="45522" # select the port

# Device Configs
NNODES=1 # number of nodes. 
NPROC_PER_NODE=8 # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total. Our experiments used 8x8=64 A100
resume=$1 # resume from checkpoint

# Task Configs
TAG="llama2-7b-tp8-mask-only-c4-singlenode" # this will be the name of output folder
DATA_INDEX_PATH=CACHE # path to the cache folder. Will generate if not exists
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/output"

# Transformer Configs
HIDEN_SIZE=4096 # hidden size
NUM_LAYERS=32 # number of layers
NUM_ATTN_HEADS=32 # number of attention heads
SEQ_LENGTH=4096 # sequence length

# Training Configs
TOKENIZER_MODEL="$PROJECT_PATH/assets/checkpoints/llama2_7b_hf/tokenizer.model" # path to the tokenizer model

TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
LR=5e-5
MIN_LR=5e-6
TRAIN_ITERS=2000 # number of iterations to train for
WARMUP_ITERS=0 #$(expr $TRAIN_ITERS \* 5 / 100)
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256

# intervals
SAVE_INTERVALS=500
LOG_INTERVALS=10
EVAL_INTERVALS=100
EVAL_ITERS=10

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ $resume -eq 0 ]; then
    LOAD="$PROJECT_PATH/output/oneshot_pruning/checkpoint/llama2-7b-tp8.sparse.nmprune.sp0.5hessian.ex0" # load the checkpoint
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load " 
else
    LOAD="$CKPT_SUBDIR/ckpt"
    EXTRA_CMD=""
fi  

TASK_CMD=" --gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5 "

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR/ckpt; mkdir -p $CKPT_SUBDIR/logs; export WANDB_API_KEY=$WANDB_API_KEY; echo Start Training

OPTIONS=" \
--untie-embeddings-and-output-weights \
--disable-bias-linear \
--no-position-embedding \
--use-rotary-position-embeddings \
--no-masked-softmax-fusion \
--swiglu \
--adam-eps 1e-5 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--no-rope-fusion \
--tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
--pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
--num-layers $NUM_LAYERS  \
--hidden-size $HIDEN_SIZE \
--num-attention-heads $NUM_ATTN_HEADS \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--make-vocab-size-divisible-by 1 \
--ffn-hidden-size 11008 --normalization RMSNorm \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters $TRAIN_ITERS   \
--lr $LR \
--min-lr $MIN_LR \
--lr-decay-style cosine \
--log-interval $LOG_INTERVALS \
--eval-iters $EVAL_ITERS \
--eval-interval $EVAL_INTERVALS \
--data-path "$DATA_BLEND"  \
--data-cache-path $DATA_INDEX_PATH \
--tokenizer-type Llama2Tokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--save-interval $SAVE_INTERVALS \
--save $CKPT_SUBDIR/ckpt \
--load $LOAD \
--split 98,2,0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--init-method-std 0.014  \
--log-num-zeros-in-grad \
--lr-warmup-iters $WARMUP_ITERS \
--exit-on-missing-checkpoint \
--no-gradient-accumulation-fusion \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn \
--bf16 \
--log-diff-mask \
--exit-signal-handler \
--exp-name $TAG \
${EXTRA_CMD} ${TASK_CMD}"

export CUDA_DEVICE_MAX_CONNECTIONS=1;

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py ${OPTIONS}