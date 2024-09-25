# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT one-shot pruning."""

import math
from functools import partial

import torch
# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import (get_args, get_num_microbatches, get_timers,
                      get_tokenizer, is_last_rank, print_rank_0)
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import (recv_forward,
                                                               send_forward)
from megatron.core.tensor_parallel.layers import (ColumnParallelLinear,
                                                  RowParallelLinear)
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module, GPTModel
from megatron.model.module import param_is_not_shared
from megatron.training import get_model, get_optimizer_param_scheduler
from megatron.utils import (average_losses_across_data_parallel_group,
                            get_ltor_masks_and_position_ids, unwrap_model)
from tasks.finetune_utils import build_data_loader

from .datasets import build_dataset
from .layerwrapper import WrappedGPT
from .optimizer import get_megatron_optimizer, OptimizerConfig
from .sparsity.core import (MagnitudePruner, HessianPruner)
from .sparsity.utils.modelutils import find_layers
from megatron.arguments import core_transformer_config_from_args


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
        config = core_transformer_config_from_args(get_args())
        print_rank_0("building GPT model ...")
        model = GPTModel(
            config,
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

    args = get_args()
    config = core_transformer_config_from_args(args)

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
    input_tensor = recv_forward(tensor_shape, config)

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)

    #output = model(tokens, position_ids, attention_mask)
    output = model(tokens, position_ids, attention_mask)
    #send_forward(output)

    send_forward(output, config)
    
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

def prune_wanda(data_loader, model, eval_metric):
    args = get_args()
    model.eval()

    layers = model.module.language_model.encoder.layers

    for i, layer in enumerate(layers):
        subset = find_layers(layer, layers=[RowParallelLinear, ColumnParallelLinear])

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        
        def add_batch(layer_name):
            def func(_, inp, out):
                wrapped_layers[layer_name].add_batch(inp[0].data, out[0].data)

            return func

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        with torch.no_grad():
            # For all the batches in the dataset.
            for iteration, batch in enumerate(data_loader):
                output = forward_step(batch, model, eval_metric)
                print(f"token {iteration} / {args.hessian_samples}")
                if iteration >= args.hessian_samples:
                    break
                
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if args.prunen != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % args.prunem == 0:
                        tmp = W_metric[:,ii:(ii+args.prunem)].float()
                        W_mask.scatter_(1, ii+torch.topk(tmp, args.prunen, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = self.return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = self.return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity)]
                    W_mask.scatter_(1, indices, True)

            if args.update_weight:
                subset[name].weight.data[W_mask] = 0  # set weights to zero 
            subset[name].mask[W_mask] = 0
            print(f"sparsity {(subset[name].mask==0).sum().item() / subset[name].mask.numel():.6f}")

        with torch.no_grad():
            # For all the batches in the dataset.
            for iteration, batch in enumerate(data_loader):
                output = forward_step(batch, model, eval_metric)
                print(f"token {iteration} / {args.hessian_samples}")
                if iteration >= args.hessian_samples:
                    break

def prune_hessian(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()
    
    layers = find_layers(model, layers=[RowParallelLinear, ColumnParallelLinear])
    if args.exclude_layers_from_prune and args.exclude_layers_from_prune > 0:
        try:
            import importlib
            ex_mod = importlib.import_module("tasks.pruning.exclude_layers")
            exclude_layers = getattr(ex_mod, f"exclude_layers_{args.exclude_layers_from_prune}")
            layers = {
                name: layer
                for name, layer in layers.items()
                if not any(sub in name for sub in exclude_layers)
            }
        except ImportError:
            print(f"Could not find exclude_layers_{args.exclude_layers_from_prune}")
            print("Pruning all layers")
    
    pruners = {}

    def add_batch(layer_name):
        """The hook function to calculate layerwise Hessian."""

        def func(_, inp, out):
            pruners[layer_name].add_batch(inp[0].data, out[0].data)

        return func

    for name, layer in layers.items():
        if "output_layer" in name:
            continue
        pruners[name] = HessianPruner(args, name, layer)

    if args.hessian_compute:
        handles = []
        for name in pruners:
            handles.append(layers[name].register_forward_hook(add_batch(name)))

        print("calculating layerwise Hessian")
        with torch.no_grad():
            # For all the batches in the dataset.
            for iteration, batch in enumerate(data_loader):
                output = forward_step(batch, model, eval_metric)
                print(f"token {iteration} / {args.hessian_samples}")
                if iteration >= args.hessian_samples:
                    break

        for h in handles:
            h.remove()

    for name, pruner in pruners.items():
        print(f"Pruning {name}.")
        sparsity = args.sparsity
        if args.target_layer is None:
            pruner.prune(args.sparse_pattern, args.row_b, args.col_b, sparsity, args.prunen, args.prunem)
        elif f".{args.target_layer}" in name:
            pruner.prune(args.sparse_pattern, args.row_b, args.col_b, sparsity, args.prunen, args.prunem)
        pruner.free()
    
def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    # start evaluation
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

# gradient pruning
def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def compute_gradient(data_loader, model):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}

    def forward_step(data_iterator, model):
        """Forward step."""
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels)

        return output_tensor, partial(loss_func, loss_mask)
    
    optimizer = get_megatron_optimizer(model, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0)
    optimizer.zero_grad()
        
    for model_module in model:
        model_module.train()

    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    data_iterator = iter(data_loader) 

    # Forward pass.
    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()
    
    fwd_bwd_timers = timers if args.timing_log_level > 1 else None
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        dtype=args.params_dtype,
        tensor_shape=(args.seq_length, args.micro_batch_size, args.hidden_size),
        grad_scaler=optimizer.scale_loss,
        sequence_parallel=args.sequence_parallel,
        forward_only=False,
        timers=fwd_bwd_timers
    )
    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    prune_layers = list(find_layers(model[0], layers=[RowParallelLinear, ColumnParallelLinear]).keys())
    all_layers = [ln.rpartition(".")[0] for ln, _ in model[0].named_parameters()]
    
    def get_main_grads_for_grad_norm():
        grads_for_norm = {}

        for param_group in optimizer.param_groups:
            for name, param in zip(param_group['name'], param_group['params']):
                grad = param.grad
                grad_not_none = grad is not None
                is_not_shared = param_is_not_shared(param)
                is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
                if grad_not_none and is_not_shared and is_not_tp_duplicate:
                    grads_for_norm[name] = grad

        return grads_for_norm


    # Reduce gradients.
    optimizer.reduce_model_grads(args, timers)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    named_grads = get_main_grads_for_grad_norm()

    # for name, grad in named_grads.items():
        # if name not in prune_layers:
            # continue
        # grad_norm = grad.abs().sum(-1)
        # named_grads[name] = grad_norm
        
    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()
    
    return named_grads

def prune_magnitude(model):
    args = get_args()
    pruners = {}
    layers = find_layers(model[0], layers=[RowParallelLinear, ColumnParallelLinear])

    for name, layer in layers.items():
        if "output_layer" in name:
            continue
        pruners[name] = MagnitudePruner(args, name, layer)

    for name, pruner in pruners.items():
        sparsity = args.sparsity
        if args.target_layer is None:
            print(f"Pruning {name}.")
            pruner.prune(args.sparse_pattern, args.row_b, args.col_b, sparsity, args.prunen, args.prunem)
        elif f".{args.target_layer}" in name:
            print(f"Pruning {name}.")
            pruner.prune(args.sparse_pattern, args.row_b, args.col_b, sparsity, args.prunen, args.prunem)

def prune_and_print_results(task, data_loader, calibration_dataloader, model, eval_metric, sparse_method="hessian", save_fn=lambda: None):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    if sparse_method == "hessian":
        print("Run Hessian pruning...")
        prune_hessian(calibration_dataloader, model[0], eval_metric)
    elif sparse_method == "magnitude":
        print("Run magnitude pruning...")
        prune_magnitude(model)
    elif sparse_method == "wanda":
        print("Run wanda pruning...")
        prune_wanda(calibration_dataloader, model[0], eval_metric)

    save_fn()
    
    output = evaluate(data_loader, model[0], eval_metric)

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


def main():
    """Main program."""
    args = get_args()
    args.prunen = 2
    args.sparsity=0.5
    if args.update_weight:
        args.save = args.save+'.update_weight'

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.task == "LAMBADA":
        eval_metric = "accuracy"
    elif args.task == "PRUNE-WIKITEXT103" or args.task == "PRUNE-WIKITEXT2":
        eval_metric = "loss"
    else:
        raise NotImplementedError("{} task is not implemented.".format(args.task))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=True if args.sparse_method == "gradient" or args.sparse_method == "taylor" else False)

    if args.load is not None:
        iteration, num_floating_point_operations_so_far = load_checkpoint(model, None, None)
        consumed_train_samples = args.consumed_train_samples
        consumed_valid_samples = args.consumed_valid_samples

    for model_module in model:
        for name, m in model_module.named_modules():
            if 'output_layer' not in name and hasattr(m, 'add_mask'):
                print("Adding sparse masks for module {}".format(name))
                m.add_mask()

    assert len(model) == 1, "Above condition should have caught this"

    # Data stuff.
    calibration_dataset = build_dataset(args.calibration_set)
    calibration_dataloader = build_data_loader(
        calibration_dataset, args.micro_batch_size, args.num_workers, drop_last=False
    )
    print("Len(calibration_dataset): ", len(calibration_dataset))
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(
        dataset, args.micro_batch_size, args.num_workers, drop_last=False
    )
    print("Len(test_dataset): ", len(dataset))

    def save_fn():
        if args.save:
            args.consumed_train_samples = consumed_train_samples
            args.consumed_valid_samples = consumed_valid_samples
            save_checkpoint(iteration, model, None, None, num_floating_point_operations_so_far)

    # Run evaluation.
    prune_and_print_results(args.task, dataloader, calibration_dataloader, model, eval_metric, args.sparse_method, save_fn=save_fn)

    print_rank_0("done :-)")