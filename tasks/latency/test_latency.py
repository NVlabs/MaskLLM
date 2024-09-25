# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT zero-shot evaluation."""

import math

import torch

from megatron import get_args
from megatron import print_rank_0, is_last_rank
from megatron import get_tokenizer
from megatron.core import parallel_state, tensor_parallel
from megatron.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.core.models.gpt import GPTModel
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward
from megatron.arguments import core_transformer_config_from_args
from tasks.finetune_utils import build_data_loader
import os
import torch
from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import get_tokenizer
from megatron.arguments import core_transformer_config_from_args
from tools.umct.lana_code.supernets.supernet_helper import SupernetClass
from tools.umct.lana_code.pretrainer import PretrainerClass
import os, json
import megatron
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import import_module
from .datasets import build_dataset

import learnable_sparsity

def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()
        config = core_transformer_config_from_args(args)

        if eval_metric == 'loss':
            parallel_output = True
        elif eval_metric == 'accuracy':
            parallel_output = False
        else:
            raise NotImplementedError('output type for {} evaluation metric '
                                      'is not supported.'.format(eval_metric))

        print_rank_0('building GPT model ...')
        if args.use_mcore_models:
            if args.spec is not None:
                transformer_layer_spec = import_module(args.spec)
            else:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
            )
        else:
            assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"
            model = megatron.model.GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )

        if not args.disable_lana:
            learnable_sparsity.convert_to_sparse_model(
                model, 
                hard=args.hard,
                N=args.N,
                M=args.M,
                temperature=args.gumbel_temperature_range,
                scale_multiplier=args.gumbel_scale_range,
                factorized=args.factorized,
                exclude=[model.language_model.output_layer],
                freeze_weight=args.mask_only,
            )

        if args.freeze_mask:
            for name, m in model.named_modules():
                if 'output_layer' not in name and hasattr(m, 'add_mask'):
                    print("Adding sparse masks for module {}".format(name))
                    m.add_mask()
        print_rank_0(model)
        return model

    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().cuda().contiguous().byte()
    tokens_ = batch['text'].long().cuda().contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, attention_mask, position_ids, loss_mask


def forward_step(batch, model, eval_metric, config):
    """Forward step."""
    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = batch
    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)
    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    input_tensor = recv_forward(tensor_shape, config)
    # Forward pass through the model.
    unwrapped_model = unwrap_model(model)
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)
    #send_forward(output, config)
    return None


def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    for iteration, batch in enumerate(data_loader):
        break
    
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(batch)
    batch = (tokens, labels, attention_mask, position_ids, loss_mask)
    
    # 20 iterations for warmup.

    with torch.no_grad():
        for _ in range(20):
            
            output = forward_step(batch, model, eval_metric, config)
        # Measure the latency.
        
        latency = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = forward_step(batch, model, eval_metric, config)
            end.record()
            torch.cuda.synchronize()
            latency.append(start.elapsed_time(end))
        latency = torch.tensor(latency)
        latency_mu, latency_std = latency.mean().item(), latency.std().item()
        # latency in ms
        print("Latency: ", latency_mu, latency_std, flush=True)
        mem = torch.cuda.max_memory_allocated()
        print("Peak Mem: ", mem, flush=True)
    return None


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""
    print("len(data_loader):", len(data_loader))
    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    eval_metric = 'loss'
    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=False)
    print_rank_0(model)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    for name, module in model[0].named_modules():
        if hasattr(module, 'weight'):
            weight_sparsity = torch.sum(module.weight == 0).item() / module.weight.numel()
            print_rank_0(f"[sparsity] {name}.weight: {weight_sparsity:.2f}")
        if hasattr(module, 'mask'):
            mask_sparsity = torch.sum(module.mask == 0).item() / module.mask.numel()
            print_rank_0(f"[sparsity] {name}.mask: {mask_sparsity:.2f}")

    if args.semi_structured:
        for name, m in model[0].named_modules():
            if 'output_layer' not in name and hasattr(m, 'add_mask'):
                print("To semi-structured sparsity {}".format(name))
                m.to_sparse_semi_structured()

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Data stuff.
    dataset = build_dataset('WIKITEXT103')
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)

    # Run evaluation.
    evaluate_and_print_results('WIKITEXT103', dataloader, model, eval_metric)

    print_rank_0('done :-)')
