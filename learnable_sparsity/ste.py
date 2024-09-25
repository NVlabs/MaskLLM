import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter
import warnings
from typing import Any, Callable, Optional

from megatron import get_args
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, get_tensor_model_parallel_world_size, _initialize_affine_weight_cpu, _initialize_affine_weight_gpu, linear_with_grad_accumulation_and_async_allreduce, set_tensor_model_parallel_attributes, _grad_accum_fusion_available, linear_with_frozen_weight
from megatron.core.tensor_parallel.utils import (
    divide,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.utils import divide
from megatron import print_rank_0

from megatron.model.utils import init_method_normal, scaled_init_method_normal


class Sparse(torch.autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device, dtype=weight_temp.dtype)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None


class ColumnParallelLinearSparse(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available to all GPUs, otherwise, every GPU will have its output which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be set to False. It returns the master weights used for initialization.
        skip_bias_add: If True, do not add the bias term, instead return it to be added by the caller. This enables performance optimations where bias can be fused with other elementwise operations.
        skip_weight_param_allocation: If True, weight parameter is not allocated and must be passed as a keyword argument `weight` during the forward pass. Note that this does not affect bias, which will be allocated if bias is True. Defaults to False.
        is_expert: If True, the layer is treated as an MoE expert layer.
        config: ModelParallelConfig object
        tp_comm_buffer_name: Communication buffer name is not used in non-Transformer-Engine modules.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used

        N=2, M=4, hard=False, temperature=[4, 0.1], scale_multiplier=[1e3, 1e4], freeze_weight=False, freeze_mask=False
    ):
        super(ColumnParallelLinearSparse, self).__init__()

       # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.stride = stride

        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.config = config
        self.skip_weight_param_allocation = skip_weight_param_allocation
        
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                    )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        expert_parallel=(self.is_expert and self.expert_parallel),
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.weight = None

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter('bias', None)

        self.async_tensor_model_parallel_allreduce = (
            config.async_tensor_model_parallel_allreduce and world_size > 1
        )

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel` "
                "cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel or self.expert_parallel
        )
        

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        #################################################
        # Learnable Sparsity
        self.N = N
        self.M = M
        self.hard = hard
        self._freeze_weight = freeze_weight
        self._freeze_mask = freeze_mask
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        self.sparse=True
        # Keep a copy of original weight
        args = get_args()

        #self.weight_reg = 0
        #if args.weight_reg>0:
        #    self.register_buffer("original_weight", torch.zeros_like(self.weight))
        #    self.weight_reg = args.weight_reg

        # Create mask
        #self.register_buffer("mask", torch.zeros_like(self.weight))
        #self.diff_mask = DifferentiableMask(
        #    target_param=self.weight, 
        #    N=self.N, 
        #    M=self.M, 
        #    hard=self.hard,
        #    temperature=self.temperature, 
        #    scale_multiplier=self.scale_multiplier)
    
        #if self._freeze_weight:
        ##    weight_data = self.weight.data
        ##    del self.weight
        #    self.register_buffer("weight", weight_data)

    def freeze_mask(self):
        with torch.no_grad():
            self._freeze_mask = True
            self.diff_mask.eval()
            self.mask = self.diff_mask()

    def init_diff_mask_from_prior(self, strength=0.0):
        #if self.weight_reg > 0:
        #    self.original_weight.data = self.weight.data.clone()
        #    print_rank_0("Copy the original weight for weight regularization")

        if strength>0.0:
            with torch.no_grad():
                sparsity = (self.mask==0).sum().item() / self.mask.numel()
                print_rank_0(f"initializing mask with prior (strength={strength}) Prior Sparsity: {sparsity}")
                priors = (self.diff_mask.mask_options * self.mask.view(-1,1,4)).sum(dim=2)
                self.diff_mask.gate.data += (priors-1)*self.diff_mask.gate.std() * strength
        print_rank_0(f"max={self.diff_mask.gate.max()}, min={self.diff_mask.gate.min()}, median={self.diff_mask.gate.median()}, mean={self.diff_mask.gate.mean()}, std={self.diff_mask.gate.std()}")
        print_rank_0("\n")

    def __repr__(self):
        return f"ColumnParallelLinearSparse(input_size={self.input_size}, output_size={self.output_size}, bias={self.bias is not None}, sequence_parallel={self.sequence_parallel}, async_tensor_model_parallel_allreduce={self.async_tensor_model_parallel_allreduce}, N={self.N}, M={self.M}, freeze_weight={self._freeze_weight}, freeze_mask={self._freeze_mask}, temperature={self.temperature}, scale_multiplier={self.scale_multiplier}, hard={self.hard})"

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """

        #mask = self.diff_mask()
        ##with torch.no_grad():
        #    self.diff_mask.mask_difference = (mask - self.mask).abs().mean()
        #self.mask = mask

        #if self.weight_reg > 0:
        #    self.weight_reg_loss = (self.weight - self.original_weight).pow(2).sum()
        
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
            )

        
        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"
        
        bias = self.bias if not self.skip_bias_add else None

        if (
            self.async_tensor_model_parallel_allreduce
            or self.sequence_parallel
            or self.explicit_expert_comm
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
            
        new_weight = Sparse.apply(weight, self.N, self.M)
    
        self.sparse_weight_norm = (new_weight.detach()).pow(2).sum()

        output_parallel = torch.matmul(input_parallel, new_weight.t())

        if self.gather_output:
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def sharded_state_dict(self, prefix='', sharded_offsets=()):
        """ Sharding along axis 0, bias sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None
    
class RowParallelLinearSparse(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already split across the GPUs and we do not split again.
        init_method: method to initialize weights. Note that bias is always set to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be set to False. It returns the master weights used for initialization.
        skip_bias_add: If True, do not add the bias term, instead return it to be added by the caller. This enables performance optimations where bias can be fused with other elementwise operations.
        is_expert: If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name: Communication buffer name. Not used in
                             non-Transformer-Engine modules.
        config: ModelParallelConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used

        N=2, M=4, hard=False, temperature=[4, 0.1], scale_multiplier=[1e3, 1e4], freeze_weight=False, freeze_mask=False
    ):
        super(RowParallelLinearSparse, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel or self.expert_parallel
        )

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        #################################################
        # Learnable Sparsity
        self.N = N
        self.M = M
        self.hard = hard
        self._freeze_weight = freeze_weight
        self._freeze_mask = freeze_mask
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        self.sparse=True

        # Keep a copy of original weight
        args = get_args()
        #self.weight_reg = 0
        #if args.weight_reg>0:
        #    self.register_buffer("original_weight", torch.zeros_like(self.weight))
        #    self.weight_reg = args.weight_reg
            
        # Create mask
        #self.register_buffer("mask", torch.zeros_like(self.weight))
        #self.diff_mask = DifferentiableMask(
        #    target_param=self.weight, 
        #    N=self.N, 
        #    M=self.M, 
        #    hard=self.hard,
        #    temperature=self.temperature, 
        #    scale_multiplier=self.scale_multiplier)
        #    
        #if self._freeze_weight:
        #    weight_data = self.weight.data
        #    del self.weight
        #    self.register_buffer("weight", weight_data)

    def freeze_mask(self):
        with torch.no_grad():
            self._freeze_mask = True
            self.diff_mask.eval()
            self.mask = self.diff_mask()

    def init_diff_mask_from_prior(self, strength=0.0):
        #if self.weight_reg > 0:
        #    self.original_weight.data = self.weight.data.clone()
        #    print_rank_0("Copy the original weight for weight regularization")

        if strength>0.0:
            with torch.no_grad():
                sparsity = (self.mask==0).sum().item() / self.mask.numel()
                print_rank_0(f"initializing mask with prior (strength={strength}) Prior Sparsity: {sparsity}")
                priors = (self.diff_mask.mask_options * self.mask.view(-1,1,4)).sum(dim=2)
                self.diff_mask.gate.data += (priors-1)*self.diff_mask.gate.std() * strength
        print_rank_0(f"max={self.diff_mask.gate.max()}, min={self.diff_mask.gate.min()}, median={self.diff_mask.gate.median()}, mean={self.diff_mask.gate.mean()}, std={self.diff_mask.gate.std()}")
        print_rank_0("\n")

    def __repr__(self):
        return f"RowParallelLinearSparse(input_size={self.input_size}, output_size={self.output_size}, bias={self.bias is not None}, input_is_parallel={self.input_is_parallel}, sequence_parallel={self.sequence_parallel}, N={self.N}, M={self.M}, freeze_weight={self._freeze_weight}, freeze_mask={self._freeze_mask}, temperature={self.temperature}, scale_multiplier={self.scale_multiplier}, hard={self.hard})"
    
    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"
        
        #if self.weight_reg > 0:
        #    self.weight_reg_loss = (self.weight - self.original_weight).pow(2).sum()

       #if not self._freeze_mask:
       #    mask = self.diff_mask()
       #    with torch.no_grad():
       #        self.diff_mask.mask_difference = (mask - self.mask).abs().mean()
       #    self.mask = mask

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        new_weight = Sparse.apply(self.weight, self.N, self.M)
        #self.sparse_weight_norm = (self.weight.detach() * self.mask).pow(2).sum()

        output_parallel = torch.matmul(input_parallel, new_weight.t())
        
        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def sharded_state_dict(self, prefix='', sharded_offsets=()):
        """ Sharding along axis 1, bias not sharded """
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None

from functools import partial
class DifferentiableMask(nn.Module):
    def __init__(
            self, 
            target_param, 
            N=2, 
            M=4, 
            hard=False,
            temperature=3.0, 
            scale_multiplier=[1e3, 1e4], 
        ):
        '''
        Implementation of differantiable mask learner
        args:
            temperature: temperature of the gumbel distribution
            gate_param: the parameter to be masked
            init_prob: initial probability of the mask
            scale_multiplier: multiplies learned gates by this value, it is needed to make the gates more sensitive to small learning rates
            initialization: "none" means we start from 0.95 for all and dont bother with initialization, "initial_mask" means we start from the initial mask
            hard: sampling mask for full gumbel
            
        temperature parameter needs more attention, we should do annealing it during training
        '''
        super().__init__()
        self.N = N
        self.M = M
        self.args = get_args() # for scheduling
        self.mask_difference = 1.0
        self.initial_mask_size = target_param.size()
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        
        # Gumbel parameters
        init_partial = partial(init.normal_, std=0.01)
        #init_partial = partial(init.constant_, val=0.01)
        if self.N==2 and self.M==4:
            print_rank_0("initalizing mask options for 2:4...")
            self.gate = Parameter(torch.empty(
                target_param.numel()//4, 6,
                device=torch.cuda.current_device(), dtype=torch.float32))
            _initialize_affine_weight_gpu(self.gate, init_partial, partition_dim=1, stride=1)
            mask_options = torch.zeros(1, 6, 4, 
                                    device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 0, :].data += torch.tensor([1, 1, 0, 0], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 1, :].data += torch.tensor([1, 0, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 2, :].data += torch.tensor([1, 0, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 3, :].data += torch.tensor([0, 1, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 4, :].data += torch.tensor([0, 1, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 5, :].data += torch.tensor([0, 0, 1, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        elif self.N==1 and self.M==4:
            print_rank_0("initalizing mask options for 1:4...")
            self.gate = Parameter(torch.empty(
                target_param.numel()//4, 4,
                device=torch.cuda.current_device(), dtype=torch.float32))
            _initialize_affine_weight_gpu(self.gate, init_partial,
                                        partition_dim=1, stride=1)
            mask_options = torch.zeros(1, 4, 4, 
                                    device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 0, :].data += torch.tensor([1, 0, 0, 0], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 1, :].data += torch.tensor([0, 1, 0, 0], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 2, :].data += torch.tensor([0, 0, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
            mask_options[:, 3, :].data += torch.tensor([0, 0, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        else:
            raise NotImplementedError
        
        self.register_buffer("mask_options", mask_options)
        self.hard = hard

        self.current_scale_multiplier = self.scale_multiplier[0]
        self.current_temperature = self.temperature[0]
        self.current_max_prob = 0.0

    

    def forward(self): 
    
        if self.training:
            start_temp, end_temp = self.temperature 
            self.current_temperature = start_temp + (end_temp - start_temp) * (self.args.iteration / self.args.train_iters)
            start_scale, end_scale = self.scale_multiplier
            self.current_scale_multiplier = start_scale + (end_scale - start_scale) * (self.args.iteration / self.args.train_iters)
            
            sampling_tensor = self.gate * self.current_scale_multiplier
            choices = gumbel_softmax(sampling_tensor, tau=self.current_temperature, hard=self.hard)
            self.current_max_prob = choices.max(-1)[0].mean().item()
            backprop_gate = (choices.unsqueeze(1) @ self.mask_options).squeeze(1)
            backprop_gate = backprop_gate.reshape(self.initial_mask_size)
        else:
            # just based on the maximum logit
            backprop_gate = self.mask_options[torch.arange(self.mask_options.shape[0]), self.gate.argmax(dim=-1)]
            backprop_gate = backprop_gate.reshape(self.initial_mask_size)
        self.sampled_gate = backprop_gate
        return backprop_gate


def _get_sparse_column_parallel_linear_layer(parent, hard=False, N=2, M=4, temperature=[4, 0.1], scale_multiplier=[1e3, 1e4], bias=False, freeze_mask=False, freeze_weight=False):
    assert isinstance(parent, ColumnParallelLinear)
    args = get_args()
    init_method = init_method_normal(args.init_method_std)
    layer = ColumnParallelLinearSparse(
                parent.input_size,
                parent.output_size,
                config=parent.config,
                init_method=init_method,
                bias=bias,
                gather_output=parent.gather_output,
                skip_bias_add=parent.skip_bias_add,
                is_expert=parent.is_expert,
                hard=hard, N=N, M=M, temperature=temperature, scale_multiplier=scale_multiplier, freeze_mask=freeze_mask, freeze_weight=freeze_weight
            )
    return layer

def _get_sparse_row_paralel_linear_layer(parent, hard=False, N=2, M=4, temperature=[4, 0.1], scale_multiplier=[1e3, 1e4], bias=False, freeze_mask=False, freeze_weight=False):
    assert isinstance(parent, RowParallelLinear)
    args = get_args()
    scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
    layer = RowParallelLinearSparse(
                parent.input_size,
                parent.output_size,
                config=parent.config,
                bias=bias,
                input_is_parallel=parent.input_is_parallel,
                init_method=scaled_init_method,
                skip_bias_add=parent.skip_bias_add,
                hard=hard, N=N, M=M, temperature=temperature, scale_multiplier=scale_multiplier, freeze_mask=freeze_mask, freeze_weight=freeze_weight
            )
    return layer

def convert_to_sparse_model(model, hard, N, M, temperature, scale_multiplier, bias=False, exclude=[], freeze_mask=False, freeze_weight=False):
    """replace linear layers with sparse linear layers"""
    for name, module in model.named_children():
        if module in exclude:
            continue
        if isinstance(module, ColumnParallelLinear):
            setattr(model, name, _get_sparse_column_parallel_linear_layer(module, hard=hard, N=N, M=M, temperature=temperature, scale_multiplier=scale_multiplier, bias=bias, freeze_mask=freeze_mask, freeze_weight=freeze_weight))
        elif isinstance(module, RowParallelLinear):
            setattr(model, name, _get_sparse_row_paralel_linear_layer(module, hard=hard, N=N, M=M, temperature=temperature, scale_multiplier=scale_multiplier, bias=bias, freeze_mask=freeze_mask, freeze_weight=freeze_weight))
        else:
            convert_to_sparse_model(module, hard=hard, N=N, M=M, temperature=temperature, scale_multiplier=scale_multiplier, bias=bias, exclude=exclude, freeze_mask=freeze_mask, freeze_weight=freeze_weight)
    return model