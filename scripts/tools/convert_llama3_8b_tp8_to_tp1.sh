#!/bin/bash
PROJECT_DIR=$(pwd)

TP=1
TP8_DIR=$PROJECT_DIR/output/checkpoints/llama3-8b-tp8-mask-only-c4-singlenode/train_iters_2000/ckpt
TP1_DIR=$PROJECT_DIR/output/checkpoints/llama3-8b-tp1-mask-only-c4-singlenode/train_iters_2000/ckpt
TOKENIZER_MODEL=assets/checkpoints/llama3_8b_hf/tokenizer.model

OPTIONS=" \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --target-tensor-parallel-size ${TP} \
    --load-dir ${TP8_DIR} \
    --save-dir ${TP1_DIR} \
    --megatron-path ${PROJECT_DIR}"

echo $TP8_DIR
echo $TP1_DIR

pip install transformers wandb accelerate tqdm; cd $PROJECT_DIR/tools/checkpoint; python util.py $OPTIONS
