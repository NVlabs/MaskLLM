# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT one-shot pruning."""

import copy
import json
import math
import os
from collections import OrderedDict

import torch
# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args, get_tokenizer, is_last_rank, print_rank_0
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.core import parallel_state, tensor_parallel
from megatron.core.pipeline_parallel.p2p_communication import (recv_forward,
                                                               send_forward)
from megatron.core.tensor_parallel.layers import (ColumnParallelLinear,
                                                  RowParallelLinear)
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module, GPTModel
from megatron.model.transformer import ParallelMLP
from megatron.optimizer import get_megatron_optimizer
from megatron.training import get_model, get_optimizer_param_scheduler
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from tasks.finetune_utils import build_data_loader

from .datasets import build_dataset
from .sparsity.core import Pruner
from .sparsity.utils.modelutils import find_layers


def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        if eval_metric == "loss":
            parallel_output = True
        elif eval_metric == "accuracy":
            parallel_output = False
        else:
            raise NotImplementedError(
                "output type for {} evaluation metric "
                "is not supported.".format(eval_metric)
            )

        print_rank_0("building GPT model ...")
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=parallel_output,
            pre_process=pre_process,
            post_process=post_process,
        )

        return model

    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch["pad_mask"].long().cuda().contiguous().byte()
    tokens_ = batch["text"].long().cuda().contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, labels, attention_mask, position_ids, loss_mask


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if parallel_state.is_pipeline_last_stage():
        # For loss, return the unreduced loss.
        if eval_metric == "loss":
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                output.contiguous().float(), labels.contiguous()
            )
            loss = torch.sum(losses.view(-1) * loss_mask.contiguous().view(-1).float())
            return loss

        # For accuracy, return the number of correctly predicted samples.
        if eval_metric == "accuracy":
            outputs = torch.argmax(output, -1)
            correct = (outputs == labels).float()
            correct[(1 - loss_mask).bool()] = 1
            correct = correct.prod(-1)
            return correct.sum()

        raise NotImplementedError(
            "forward method for evaluation metric {} "
            "is not implemented.".format(eval_metric)
        )
    return None

def build_pruner(model, dataloader, eval_metric):
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    layers = find_layers(model, layers=[RowParallelLinear, ColumnParallelLinear])
    
    pruners = {}

    def add_batch(layer_name):
        """The hook function to calculate layerwise Hessian."""

        def func(_, inp, out):
            pruners[layer_name].add_batch(inp[0].data, out[0].data)

        return func

    for name, layer in layers.items():
        if "output_layer" in name:
            continue
        pruners[name] = Pruner(args, name, layer)

    if args.hessian_compute:
        handles = []
        for name in pruners:
            handles.append(layers[name].register_forward_hook(add_batch(name)))

        print("calculating layerwise Hessian")
        with torch.no_grad():
            # For all the batches in the dataset.
            for iteration, batch in enumerate(dataloader):
                forward_step(batch, model, eval_metric)
                print(f"token {iteration} / {args.hessian_samples}")
                if iteration >= args.hessian_samples:
                    break
        for name, pruner in pruners.items():
            pruner.save_hessian()

        for h in handles:
            h.remove()
    
    return pruners

def prune(dataloader, model, eval_metric):
    args = get_args()
    pruners = build_pruner(model, dataloader, eval_metric)
    
    for name, pruner in pruners.items():
        print(f"Pruning {name}.")
        sparsity = args.sparsity
        pruner.prune(args.sparse_pattern, args.row_b, args.col_b, sparsity)

        pruner.free()

    model.eval()
    # start evaluation
    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(dataloader):
            if iteration % args.log_interval == 0:
                print_rank_0("> working on iteration: {}".format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(
                    output, group=parallel_state.get_data_parallel_group()
                )

                total_output += output

    return total_output

def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(output,
                                             group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output

def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = " validation results on {} | ".format(task)
    if is_last_rank():
        if eval_metric == "loss":
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += "avg loss: {:.4E} | ".format(val_loss)
            string += "ppl: {:.4E} | ".format(ppl)
            string += "adjusted ppl: {:.4E} | ".format(adjusted_ppl)
            string += "token ratio: {} |".format(token_ratio)

        elif eval_metric == "accuracy":
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += "number correct: {:.4E} | ".format(output)
            string += "total examples: {:.4E} | ".format(num_examples)
            string += "avg accuracy: {:.4E}".format(acc)

        else:
            raise NotImplementedError(
                "evaluation method for {} metric is not "
                "implemented yet.".format(eval_metric)
            )

        length = len(string) + 1
        print("-" * length)
        print(string)
        print("-" * length)

def save_sensitivity(sensitivities, fname="sensitivity_results.json"):
    args = get_args()
    json_obj = json.dumps(sensitivities)

    print(sensitivities)
    json_file = os.path.join(args.save, fname)
    with open(json_file, "w") as f:
        f.write(json_obj)

def load_sensitivity(fname="sensitivity_results.json"):
    args = get_args()
    json_file = os.path.join(args.save, fname)
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            json_obj = json.load(f)
            return json_obj
    else:
        return None

def set_submodule(model, target, target_submodule):
    """The set function that complements nn.Module.get_submodule()."""
    assert target != "", "Cannot set root module"

    # Verify the original submodule exists
    model.get_submodule(target)
    parent_module = model.get_submodule(target.rpartition(".")[0])
    child_name = target.split(".")[-1]
    parent_module.add_module(child_name, target_submodule)

def sensitivity_analysis(model, data_loader, sparsities, eval_metric):
    args = get_args()
    pruners = build_pruner(model, data_loader, eval_metric)
    output = evaluate(data_loader, model, eval_metric) / (data_loader.dataset.num_tokenized_tokens - 1)
    
    sensitivities = load_sensitivity()
    if sensitivities is None:
        sensitivities = {"baseline": {"1.0": float(output)}}

    for layer_name, base_pruner in pruners.items():
        if layer_name in sensitivities:
            continue

        sensitivity = {}
        for sparsity in sparsities:
            pruner = copy.deepcopy(base_pruner)
            pruner.prune(args.sparse_pattern, args.row_b, args.col_b, sparsity[2], sparsity[0], sparsity[1])
            dense_module = model.get_submodule(pruner.layer_name)
            set_submodule(model, pruner.layer_name, pruner.layer)
            output = evaluate(data_loader, model, eval_metric) / (data_loader.dataset.num_tokenized_tokens - 1)
            # num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            # val_loss = output / (num_tokenized_tokens - 1)
            set_submodule(model, pruner.layer_name, dense_module)

            sensitivity[sparsity[2]] = float(output)
            sensitivities[layer_name] = sensitivity

        save_sensitivity(sensitivities)
    return sensitivities

def prune_and_print_results(data_loader, model, eval_metric):
    """Evaluate and print results on screen."""
    sensitivity_analysis(model, data_loader, [(1, 4), (2, 4), (3, 4)], eval_metric)
    # output = prune(data_loader, model, eval_metric)


def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.task == "LAMBADA":
        eval_metric = "accuracy"
    elif args.task == "WIKITEXT103" or args.task == "PRUNE-WIKITEXT103":
        eval_metric = "loss"
    else:
        raise NotImplementedError("{} task is not implemented.".format(args.task))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=False)
    optimizer = get_megatron_optimizer(model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None:
        load_checkpoint(model, optimizer, opt_param_scheduler)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Data stuff.
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(
        dataset, args.micro_batch_size, args.num_workers, drop_last=False
    )

    # Run sensitivity analysis.
    sensitivity_analysis(model, dataloader, [(1, 4, 0.25), (2, 4, 0.5), (3, 4, 0.75)], eval_metric)

    print_rank_0("done :-)")
