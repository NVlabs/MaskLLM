# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""
Train N:M Sparsity for GPTs (General GPTs / Nemotron / LLaMA)
adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py
"""

import os
import torch
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group, 
    unwrap_model
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import mpu

import learnable_sparsity
import os

def load_partial_state_dict(self, state_dict, strict=True):
    """
    # This function is used to load the LLM weights from the checkpoint
    # For those layers with method ``init_diff_mask_from_prior'', 
    # it intialize ``layer.diff_mask.gate'' with the prior mask ``layer.mask''
    """
    if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
        self.word_embeddings.load_state_dict(
            state_dict[self._word_embeddings_for_head_key], strict=strict)
    if self._language_model_key in state_dict:
        state_dict = state_dict[self._language_model_key]
        
    self.language_model.load_state_dict(state_dict, strict=False) #

    args = get_args()
    # Initialize the diff mask from the .mask prior if needed
    for name, m in self.language_model.named_modules():
        if hasattr(m, 'init_diff_mask_from_prior') and hasattr(m, 'mask'):
            m.init_diff_mask_from_prior(args.prior_strength)


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    assert args.use_mcore_models==False, "Megatron mcore is supported for sparsity"
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

    # Replace Linear layers with Sparse Linear layers
    excluded_layers = []
    if hasattr(model.language_model, 'output_layer'):
        excluded_layers.append(model.language_model.output_layer) # always skip the output linear layer
    learnable_sparsity.convert_to_sparse_model(
        model, 
        hard=args.hard, # By default, we use soft mask for training, which yields better perforamnce than hard masks
        N=args.N, # number of non-zero parameters in N:M sparsity
        M=args.M, # block size in N:M sparsity
        temperature=args.gumbel_temperature_range, # Annealing temperature for Gumbel softmax, default [4, 0.05] with linear scheduler
        scale_multiplier=args.gumbel_scale_range, # Scale multiplier for Gumbel logits, default [1e2, 5e2] with linear scheduler
        exclude=excluded_layers,
        freeze_weight=args.mask_only, # Freeze the weights of the sparse layers, it will transform .weight to a buffer with .register_buffer
    )

    if args.enable_partial_load: # Loading LLM weights from a dense LLM checkpoint
        print("Replacing load_state_dict function for partial loading", flush=True)
        new_load_state_dict_function = load_partial_state_dict.__get__(model, model.__class__)
        setattr(model, 'load_state_dict', new_load_state_dict_function)

    model.sparse_linears = [] # record all sparse linear layers for weight regularization 
    for name, m in model.language_model.named_modules():
        if hasattr(m, 'diff_mask'):
            model.sparse_linears.append(m)
    print_rank_0(f"Found {len(model.sparse_linears)} sparse layers")

    print_rank_0(model)
    return model

def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()

def loss_func(loss_mask: torch.Tensor, model, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    loss_reg = torch.zeros_like(loss)
    if args.weight_reg>0:
        unwrapped_models = unwrap_model(model)
        for m in unwrapped_models.sparse_linears:
            loss_reg += m.sparse_weight_norm # maximize the norm of the remaining weights
    weight_loss_reg = args.weight_reg * (1 - args.iteration / args.train_iters) # linearly remove the weight regularization term

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )
    averaged_loss, averaged_loss_reg = average_losses_across_data_parallel_group([loss, loss_reg])

    # maximize the norm of the sparse weights
    return loss * args.context_parallel_size - weight_loss_reg * loss_reg, {'lm loss': averaged_loss, 'reg loss': averaged_loss_reg}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask, model)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        vocab_size=get_tokenizer().vocab_size,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step)
