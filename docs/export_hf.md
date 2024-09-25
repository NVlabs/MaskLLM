# Export to Hugging Face
Here we use the LLaMA-2/3 models as an example. The following pipeline will perform the converstion `Megatron with .mask (TP=8)` -> `Megatron with TP=8` -> `Megatron with TP=1` -> `Hugging Face`.

### Merge sparse masks into model parameters

This script will create a new checkpoint named as `iter_0000001` under the same folder. It will be a standard megatron checkpoints with 0s in the weights.
```bash
# Llama-2 7B
python tool_apply_sparsity.py --ckpt_dir output/checkpoints/llama2-7b-tp8-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000

# Llama-3 8B
python tool_apply_sparsity.py --ckpt_dir output/checkpoints/llama3-8b-tp8-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000
```

To load this checkpoint with Megatron, please modify the value in `latest_checkpointed_iteration.txt` as `1` for loading. 

### Convert Tensor Parallelism to 1

Then we need to convert the model from TP=8 to TP=1 to disable Tensor Parallelism.
```bash
# Convert llama2 7b from TP=8 to TP=1
bash scripts/tools/convert_llama2_7b_tp8_to_tp1.sh

# Convert llama3 8b from TP=8 to TP=1
bash scripts/tools/convert_llama3_8b_tp8_to_tp1.sh 
```

Output:
```
output/checkpoints
├── llama2-7b-tp1-mask-only-c4-singlenode # <== TP=1
├── llama2-7b-tp8-mask-only-c4-singlenode # <== TP=8
├── llama3-8b-tp1-mask-only-c4-singlenode # <== TP=1
└── llama3-8b-tp8-mask-only-c4-singlenode # <== TP=8
```

### Export to Hugging Face

```bash
# Llama-2 7B (Don't forget to copy the tokenizer)
python tool_export_to_hf.py --hf_ckpt assets/checkpoints/llama2_7b_hf --megatron_ckpt output/checkpoints/llama2-7b-tp1-mask-
only-c4-singlenode/train_iters_2000/ckpt/iter_0002000/mp_rank_00/model_optim_rng.pt --save_ckpt output/checkpoints/llama2_7b_hf_maskllm_c4
cp assets/checkpoints/llama2_7b_hf/tokenizer* output/checkpoints/llama2_7b_hf_maskllm_c4/

# Llama-3 8B
python tool_export_to_hf.py --hf_ckpt assets/checkpoints/llama3_8b_hf --megatron_ckpt output/checkpoints/llama3-8b-tp1-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000/mp_rank_00/model_optim_rng.pt --save_ckpt output/checkpoints/llama3_8b_hf_maskllm_c4 --num_query_groups 8 --group_query_attention
cp assets/checkpoints/llama3_8b_hf/tokenizer* output/checkpoints/llama3_8b_hf_maskllm_c4/
cp assets/checkpoints/llama3_8b_hf/special_tokens_map.json output/checkpoints/llama3_8b_hf_maskllm_c4/
```

### Eval the exported HF model
```bash
# Llama-2 7B
python eval_llama_ppl.py --model output/checkpoints/llama2_7b_hf_maskllm_c4/

# Llama-3 8B
python eval_llama_ppl.py --model output/checkpoints/llama3_8b_hf_maskllm_c4/
```

### Pre-compute an isolated mask file for HF model
```bash
# Llama-2 7B
python tool_compute_mask_hf.py --dense assets/checkpoints/llama2_7b_hf --sparse output/checkpoints/llama2_7b_hf_maskllm_c4 --save output/checkpoints/llama2_7b_hf_maskllm_c4_mask.pt

# Llama-3 8B
python tool_compute_mask_hf.py --dense assets/checkpoints/llama3_8b_hf --sparse output/checkpoints/llama3_8b_hf_maskllm_c4 --save output/checkpoints/llama3_8b_hf_maskllm_c4_mask.pt
```

Then we can compress the mask file with ``np.savez_compressed``.

```bash
mkdir -p output/precomputed_masks

# Llama-2 7B
python tool_compress_mask.py --mask_ckpt output/checkpoints/llama2_7b_hf_maskllm_c4_mask.pt --output assets/precomputed_masks/llama2_7b_hf_maskllm_c4_mask_compressed.npz

# Llama-3 8B
python tool_compress_mask.py --mask_ckpt output/checkpoints/llama3_8b_hf_maskllm_c4_mask.pt --output assets/precomputed_masks/llama3_8b_hf_maskllm_c4_mask_compressed.npz
```

### Evaluate the official HF model with precomputed masks
```bash
# Llama-2 7B
python eval_llama_ppl.py --model meta-llama/Llama-2-7b-hf --mask assets/precomputed_masks/llama2_7b_hf_maskllm_c4_mask_compressed.npz

# Llama-3 8B
python eval_llama_ppl.py --model meta-llama/Meta-Llama-3-8B --mask assets/precomputed_masks/llama3_8b_hf_maskllm_c4_mask_compressed.npz
```