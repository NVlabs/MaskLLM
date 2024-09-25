#!/bin/bash

# export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
LOAD=$1 # path to the model
MODEL=$2 # 7b, 13b
TP=$3
MODE=$4

echo $LOAD

PROJECT_DIR=$(pwd) # change this to the path of your maskllm project
OUTPUT="$PROJECT_DIR/output"

# If model==2b
if [ "$MODEL" == "8b" ]; then
   HIDDEN_SIZE=4096 # hidden size
   NUM_LAYERS=32 # number of layers
   NUM_ATTN_HEADS=32 # number of attention heads
   TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/llama3_8b_hf"
   FFN_HIDDEN_SIZE=14336
fi
SEQ_LENGTH=4096 # sequence length

if [ "$MODE" == "dense" ]; then
   MASK_OPTIONS=" "
elif [ "$MODE" == "sparse" ]; then
   MASK_OPTIONS="--enable-sparsity "
else
   MASK_OPTIONS=" "
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1;

OPTIONS=" \
   --task WIKITEXT2 \
   --use-flash-attn \
   --untie-embeddings-and-output-weights \
   --disable-bias-linear \
   --no-position-embedding \
   --no-masked-softmax-fusion \
   --use-rotary-position-embeddings \
   --rotary-base 500000 \
   --swiglu \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --tensor-model-parallel-size $TP \
   --pipeline-model-parallel-size 1 \
   --overlapping-eval $SEQ_LENGTH \
   --num-layers $NUM_LAYERS \
   --hidden-size $HIDDEN_SIZE \
   --num-attention-heads $NUM_ATTN_HEADS \
   --seq-length $SEQ_LENGTH \
   --max-position-embeddings $SEQ_LENGTH \
   --group-query-attention \
   --num-query-groups 8 \
   --micro-batch-size 1 \
   --global-batch-size 256 \
   --train-iters 1 \
   --lr-decay-iters 1 \
   --lr 1.0e-4 \
   --min-lr 1.0e-5 \
   --lr-decay-style cosine \
   --log-interval 100 \
   --tokenizer-type AutoTokenizer \
   --tokenizer-model ${TOKENIZER_MODEL} \
   --make-vocab-size-divisible-by 1 \
   --ffn-hidden-size $FFN_HIDDEN_SIZE --normalization RMSNorm \
   --data-path None \
   --bf16 \
   --no-save-optim --no-save-rng \
   --no-load-optim --no-load-rng \
   --exit-on-missing-checkpoint \
   --load ${LOAD} \
   --hidden-dropout 0.0 --attention-dropout 0.0 \
   $MASK_OPTIONS"

cd $PROJECT_DIR; 

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45530" # select the port
NNODES=1 # number of nodes
NPROC_PER_NODE=${TP} # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tasks/main.py ${OPTIONS}

