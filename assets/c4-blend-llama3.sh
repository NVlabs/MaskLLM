#!/bin/bash

# multilingual datasets
C4_HOME=assets/data/preprocessed_llama3
DATA_BLEND=""
for i in {00000..00019}; do # 1/20
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/llama3_${i}_text_document"
done
