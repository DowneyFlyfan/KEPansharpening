import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Iterable, Tuple, Union, Optional

import torch.distributed as dist
from torch.distributed._tensor import Replicate

from config import bargs


class Adam_mini_CNN(optim.Optimizer):
    def __init__(
        self,
        optim_groups: list,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        verbose=True,
    ):
        """
        Args:
            optim_groups: containing name, lr, params and additional info for each parameter
            betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`): Same as Adam's betas parameters (b1, b2).
            eps (`float`, *optional*, defaults to 1e-06): Same as Adam's epsilon for numerical stability.
            weight_decay (`float`, *optional*, defaults to 0.0): Decoupled weight decay to apply.
        """
        self.adam_block_names = {"bias"}

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        minor_groups = []
        for optim_group in optim_groups:
            for name, param in optim_group["params"]:
                name = name.lower()
                if not param.requires_grad:
                    continue
                if verbose:
                    print(
                        "The optimizer found the param block with name:",
                        name,
                        param.size(),
                    )
                state = {
                    "lr": optim_group["lr"],
                    "name": f"{optim_group['module_name']}_{name}",
                    "params": param,
                    "weight_decay": (
                        0.0
                        if ("norm" in name or "ln" in name or "bias" in name)
                        else weight_decay
                    ),
                }
                minor_groups.append(state)

        super().__init__(
            minor_groups, defaults=dict(beta1=betas[0], beta2=betas[1], eps=eps)
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:
                state = self.state[p]
                if any(
                    adam_block_name in name for adam_block_name in self.adam_block_names
                ):  # for bias terms
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["step"] = 0
                        state["v"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    grad = p.grad
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)

                else:  # other blocks. By default, this is for Norms.
                    if len(state) == 0:
                        block_numel = (
                            torch.tensor(p.numel()).to(torch.float32).to(bargs.device)
                        )
                        state["m"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["step"] = 0
                        state["vmean"] = torch.zeros_like(
                            torch.sum(p * p), memory_format=torch.preserve_format
                        )
                        state["block_numel"] = block_numel.item()

                    if p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(p * p))
                    else:
                        grad = p.grad
                        tmp_lr = torch.sum(grad * grad)

                    if p.grad is None:
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * (stepsize.to(state["m"].device))
                    update.mul_(lr)
                    p.add_(-update)

        return loss


class Adam_mini(optim.Optimizer):
    def __init__(
        self,
        quantity_groups: list,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        dim: int = 2048,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        verbose=True,
    ):
        """
        Arguments:
            named_parameters ('Iterable[Tuple[str, nn.Parameter]]'): Iterable of named parameters to optimize or dictionaries defining parameter groups. Usually set to model.named_parameters()

            lr (`float`, *optional*, defaults to 0.001): The learning rate to use.
            betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`): Same as Adam's betas parameters (b1, b2).
            eps (`float`, *optional*, defaults to 1e-06): Same as Adam's epsilon for numerical stability.
            weight_decay (`float`, *optional*, defaults to 0.0): Decoupled weight decay to apply.

            dim (`int`, *optional*, defaults to 2048): Dimension for hidden features. Can be left unspecified if training non-transformer models.

            n_heads (`int`, *optional*, defaults to 32): Number of attention heads. Can be left unspecified if training non-transformer models.

            n_kv_heads (`int`, *optional*, defaults to None): Number of heads for Key and Value. Or equivalently, number of query groups in Group Query Attention. Also known as "n_query_groups". If not specified, it will be equal to n_head. Can be left unspecified if training non-transformer models.

            verbose (`bool`, *optional*, defaults to True): Print all the logs if true.
        """

        # Parameters Check
        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.verbose = verbose
        self.check_block_name = True
        self.head_numel = self.dim * self.dim // self.n_heads  # Parameters for each qkv
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.dim == int(self.dim):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not self.n_heads == int(self.n_heads):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        # Embedding layer. Use one lr per token
        self.embd_names = {"embed", "embd", "wte"}
        # Output layers. Use one lr per token
        self.output_names = {"lm_head", "output", "final_layer"}
        # Query and Keys. User one lr per head
        self.wqk_names = {"k_proj", "q_proj", "wq", "wk", "query", "key"}
        # Values. Use one lr per neuron
        # it is also okay to set self.wv_names to be empty and use a single lr for the whole v. But be cautious that this will bring extra all_reduce operations
        self.wv_names = {"v_proj", "wv", "value"}
        # attn_proj. Use one lr per neuron
        self.attn_proj_names = {"o_proj", "wo", "attn.proj"}
        # MLPs. Use one lr per neuron
        self.mlp_names = {"feed_forward", "linear", "mlp"}
        # Blocks that use Adam: bias terms
        self.adam_block_names = {"bias"}

        optim_groups = []
        for quantity_group in quantity_groups:
            for name, param in quantity_group["params"]:
                name = name.lower()
                if not param.requires_grad:
                    continue
                if verbose:
                    print(
                        "The optimizer found the param block with name:",
                        name,
                        param.size(),
                    )
                state = {
                    "lr": quantity_group["lr"],
                    "name": f"{quantity_group['module_name']}_{name}",
                    "params": param,
                    "weight_decay": (
                        0.0
                        if ("norm" in name or "ln" in name or "bias" in name)
                        else weight_decay
                    ),
                }
                optim_groups.append(state)

        defaults = dict(beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    def count_block(self):
        count_embd = 0
        count_output = 0
        count_wqk = 0
        count_wv = 0
        count_attn_proj = 0
        count_mlp = 0
        for group in self.param_groups:
            name = group["name"]
            if "bias" in name:
                continue
            if any(embd_name in name for embd_name in self.embd_names):
                count_embd += 1
            if any(output_name in name for output_name in self.output_names):
                count_output += 1
            if any(wqk_name in name for wqk_name in self.wqk_names):
                count_wqk += 1
                assert (
                    self.dim * self.dim
                ) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
            if any(wv_name in name for wv_name in self.wv_names):
                count_wv += 1
            if any(attn_proj_name in name for attn_proj_name in self.attn_proj_names):
                count_attn_proj += 1
            if any(mlp_name in name for mlp_name in self.mlp_names):
                count_mlp += 1
        if self.verbose:
            print(
                f"Adam-mini found {count_embd} embedding layers, {count_output} output layers; {count_wqk} Querys and Keys;  {count_wv} Values;  {count_attn_proj} attn_proj;  {count_mlp} MLPs;"
            )

        if count_embd == 0 and self.verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the keywords in the name of your embedding layer'). "
            )
        if count_output == 0 and self.verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found. If you are training Transformers (without weight-tying), please check the name of your output layer and manually add them to 'self.output_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.output_names.add('the keywords in the  name of your output layer').  Please ignore this warning if you are using weight-tying."
            )
        if count_wqk == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found. If you are training Transformers, please check the name of your Query and Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding two additional lines of code: optimizer.wqk_names.add('the keywords in the  name of your Query' ); optimizer.wqk_names.add('the keywords in the  name of your Key'). "
            )

        if count_wv == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Value found. If you are training Transformers, please check the name of your Value in attention blocks and manually add them to 'self.wv_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.wv_names.add('the keywords in the  name of your Value' ). "
            )

        if count_attn_proj == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No attn_proj found. If you are training Transformers, please check the name of your attn_proj in attention blocks and manually add them to 'self.attn_proj_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.attn_proj_names.add('the keywords in the  name of your attn_proj' ). "
            )

        if count_mlp == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No MLP found. If you are training Transformers, please check the name of your MLP in attention blocks and manually add them to 'self.mlp_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.attn_proj_names.add('the keywords in the  name of your MLP' ). "
            )

        if (
            count_output
            + count_embd
            + count_wqk
            + count_wv
            + count_attn_proj
            + count_mlp
            == 0
        ) and self.verbose:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers."
            )

    @torch.no_grad()
    def step(self, closure=None):
        if self.check_block_name:
            self.count_block()
            self.check_block_name = False

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:
                state = self.state[p]
                if any(
                    adam_block_name in name for adam_block_name in self.adam_block_names
                ):  # for bias terms
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["step"] = 0
                        state["v"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    # ---- Typycal Process of AdamW ----
                    grad = p.grad
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)

                elif any(
                    wqk_name in name for wqk_name in self.wqk_names
                ):  # this is for query and key
                    if p.grad is None:
                        continue
                    head_numel = self.head_numel  # group["head_numel"]
                    if len(state) == 0:
                        m = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["m"] = m.view(-1, head_numel)
                        state["head_per_gpu"] = state["m"].size(
                            0
                        )  # this is head per gpu
                        state["step"] = 0
                        # NOTE: We must use `zeros_like` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # the following line is equivalent to: state["vmean"] = torch.zeros(state["head"])
                        state["vmean"] = torch.zeros_like(
                            state["m"][0 : state["head_per_gpu"], 0:1],
                            memory_format=torch.preserve_format,
                        )

                    grad = p.grad  # .to(torch.float32)
                    head_per_gpu = state["head_per_gpu"]
                    grad = grad.view(head_per_gpu, head_numel)
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(head_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                elif (
                    any(embd_name in name for embd_name in self.embd_names)
                    or any(output_name in name for output_name in self.output_names)
                    or any(wv_name in name for wv_name in self.wv_names)
                    or any(mlp_name in name for mlp_name in self.mlp_names)
                    or any(
                        attn_proj_name in name
                        for attn_proj_name in self.attn_proj_names
                    )
                ):  # For value, mlp and embed
                    if p.grad is None:
                        continue
                    # neuron_numel = group["neuron_numel"] # assume grad is a matrix by default, so do not need this
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(
                            p.grad, memory_format=torch.preserve_format
                        )  # assume grad is a matrix by default, no need to view
                        # state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format).view(-1, neuron_numel)
                        state["step"] = 0
                        state["neuron_per_gpu"] = state["m"].size(
                            0
                        )  # this is neuron per gpu
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # for standard tensor: state["vmean"] = torch.zeros(1, device=p.device)
                        # for DTensor: state["vmean"] = p.new_zeros(1)
                        # the following implementation unifies the above two lines
                        state["vmean"] = torch.zeros_like(
                            state["m"][0 : state["neuron_per_gpu"], 0:1],
                            memory_format=torch.preserve_format,
                        )

                    grad = p.grad
                    neuron_per_gpu = state["neuron_per_gpu"]
                    # grad = grad.view(neuron_per_gpu, neuron_numel) # assume grad is a matrix by default, so no need to reshape
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(neuron_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)

                else:  # other blocks. By default, this is for LayerNorms. Sometimes it is also fine to put Value here
                    if len(state) == 0:
                        block_numel = (
                            torch.tensor(p.numel()).to(torch.float32).to(bargs.device)
                        )
                        reduced = False
                        if self.world_size > 1:
                            tensor_list = [
                                torch.zeros_like(block_numel)
                                for _ in range(self.world_size)
                            ]

                            dist.all_gather(tensor_list, block_numel)
                            s = 0
                            block_numel = 0
                            for d in tensor_list:
                                if d > 0:
                                    s = s + 1
                                block_numel = block_numel + d
                            if s >= 2:
                                reduced = True

                        state["m"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["step"] = 0
                        state["reduced"] = reduced
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # For standard tensor: state["vmean"] = torch.zeros(1, device=p.device)
                        # For DTensor: state["vmean"] = p.new_zeros(1)
                        # the following implementation unifies the above two lines
                        state["vmean"] = torch.zeros_like(
                            torch.sum(p * p), memory_format=torch.preserve_format
                        )
                        state["block_numel"] = block_numel.item()
                    if p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(p * p))
                    else:
                        grad = p.grad  # .to(torch.float32)
                        tmp_lr = torch.sum(grad * grad)

                    if state["reduced"]:
                        # Force communication over GPUs when GPUs are available
                        if tmp_lr.device.type == "cpu":
                            # Move the tensor to the current GPU device
                            tmp_lr_gpu = tmp_lr.to(torch.cuda.current_device())

                            if "device_mesh" in dir(tmp_lr):
                                # when tmp_lr is a  DTensor in TorchTitan
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                # when tmp_lr is a  standard tensor
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                            # Move the result back to the CPU tensor
                            tmp_lr.copy_(tmp_lr_gpu.cpu())
                        else:
                            # Tensor is already on GPU, use NCCL backend
                            if "device_mesh" in dir(tmp_lr):
                                # when tmp_lr is a  DTensor in TorchTitan
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                # when tmp_lr is a  standard tensor
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                    if p.grad is None:
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * (stepsize.to(state["m"].device))
                    update.mul_(lr)
                    p.add_(-update)

        return loss
