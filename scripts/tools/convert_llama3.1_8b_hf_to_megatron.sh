#!/bin/bash
PROJECT_DIR=$(pwd)

TP=8
HF_FORMAT_DIR=$PROJECT_DIR/assets/checkpoints/llama3.1_8b_hf
MEGATRON_FORMAT_DIR=$PROJECT_DIR/assets/checkpoints/llama3.1_8b_megatron_tp$TP
TOKENIZER_MODEL=$HF_FORMAT_DIR/tokenizer.json

OPTIONS=" \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size ${TP} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}"

echo $HF_FORMAT_DIR
echo $MEGATRON_FORMAT_DIR
echo $TOKENIZER_MODEL

cd $PROJECT_DIR/tools/checkpoint; python util.py $OPTIONS
cp -r $HF_FORMAT_DIR/*token* $MEGATRON_FORMAT_DIR
