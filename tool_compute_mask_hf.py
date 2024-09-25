import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory

from collections import defaultdict
import fnmatch

# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        device_map="cpu"
    )
    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', type=str, help='Dense model')
    parser.add_argument('--sparse', type=str, help='Sparse model')
    parser.add_argument('--save', type=str, help='Save as')
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    args = parser.parse_args()

    # Setting seeds for reproducibilit
    with torch.no_grad():
        dense = get_llm(args.dense, args.cache_dir)
        sparse = get_llm(args.sparse, args.cache_dir)

        mask_ckpt = {}
        for (name_dense, param_dense), (name_sparse, param_sparse) in zip(dense.named_parameters(), sparse.named_parameters()):
            sparsity = (param_sparse==0).float().mean().item()
            print(f"{name_sparse} - sparsity {sparsity:.4f}")
            # Check 2:4
            if abs(sparsity-0.5)<0.0001:
                mask = (param_sparse!=0).float()
                assert torch.equal(mask * param_dense, param_sparse)
                mask_ckpt[name_sparse+'.mask'] = mask
            else:
                # assert equal of dense and sparse_weight 
                assert torch.equal(param_dense, param_sparse)

        torch.save(mask_ckpt, args.save)
        print(mask_ckpt.keys())
        print(f"Mask saved as {args.save}")

        

if __name__ == '__main__':
    main()