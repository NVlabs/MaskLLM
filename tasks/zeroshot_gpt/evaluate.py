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
        
        if args.enable_sparsity:
            for name, m in model.named_modules():
                if 'output_layer' not in name and hasattr(m, 'add_mask'):
                    print("Adding sparse masks for module {}".format(name))
                    m.add_mask()
        elif args.add_ste:
            import learnable_sparsity
            excluded_layers = []
            if hasattr(model.language_model, 'output_layer'):
                excluded_layers.append(model.language_model.output_layer) # always skip the output linear layer
            learnable_sparsity.ste.convert_to_sparse_model(
                model, 
                hard=False, # By default, we use soft mask for training, which yields better perforamnce than hard masks
                N=2, # number of non-zero parameters in N:M sparsity
                M=4, # block size in N:M sparsity
                exclude=excluded_layers,
                temperature=[1,1], # Annealing temperature for Gumbel softmax, default [4, 0.05] with linear scheduler
                scale_multiplier=[1,1],
                freeze_weight=False, # Freeze the weights of the sparse layers, it will transform .weight to a buffer with .register_buffer
            )
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
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    input_tensor = recv_forward(tensor_shape, config)

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model)
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output, config)

    if parallel_state.is_pipeline_last_stage():
        # For loss, return the unreduced loss.
        if eval_metric == 'loss':
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                output.contiguous().float(), labels.contiguous())
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            return loss

        # For accuracy, return the number of correctly predicted samples.
        if eval_metric == 'accuracy':
            outputs = torch.argmax(output, -1)
            correct = (outputs == labels).float()
            correct[(1 - loss_mask).bool()] = 1
            correct = correct.prod(-1)
            return correct.sum()

        raise NotImplementedError('forward method for evaluation metric {} '
                                  'is not implemented.'.format(eval_metric))
    return None


def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric, config)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(output,
                                             group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""
    print("len(data_loader):", len(data_loader))
    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = ' validation results on {} | '.format(task)
    if is_last_rank():
        if eval_metric == 'loss':
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += 'avg loss: {:.4E} | '.format(val_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)

        elif eval_metric == 'accuracy':
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        else:
            raise NotImplementedError('evaluation method for {} metric is not '
                                      'implemented yet.'.format(eval_metric))

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)


def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.task == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.task in ['WIKITEXT103', 'WIKITEXT2', 'C4']:
        eval_metric = 'loss'
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.task))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=False)
    print_rank_0(model)

    if args.load is not None:
        _ = load_checkpoint(model, None, None, strict=False)

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
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)

    # Run evaluation.
    evaluate_and_print_results(args.task, dataloader, model, eval_metric)

    print_rank_0('done :-)')
