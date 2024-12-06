#!/bin/bash

# multilingual datasets
C4_HOME=assets/data/c4_llama3.1_pretokenized
DATA_BLEND=""
for i in {00000..00019}; do # 1/20
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama3.1_${i}_text_document"
done
