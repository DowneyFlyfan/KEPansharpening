"""simple, minimal implementation of mamba in one file of pytorch.

glossary:
    b: batch size                       (`b` in mamba paper [1] algorithm 2)
    l: sequence length                  (`l` in [1] algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`n` in [1] algorithm 2)
    expand: expansion factor            (`e` in [1] section 3.4)
    d_in or d_inner: d * expand         (`d` in [1] algorithm 2)
    a, b, c, d: state space parameters  (see any state space representation formula)
                                        (b, c are input-dependent (aka selective, a key innovation in mamba); a, d are not)
    δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
"""

from __future__ import annotations
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

@dataclass
class ModelArgs:
    d_model: int = 8
    n_layer: int = 6
    vocab_size: int = 64
    d_state: int = 16
    expand: int = 2
    dt_rank: int = 0
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    device: str = 'mps'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 0:
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x

        return output

class MambaBlock(nn.Module):
    def __init__(self, args = ModelArgs()):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l] # Cut off redundant parts
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float()).to(self.args.device)  # shape (d_in, n), for numerical stability
        D = self.D.float().to(self.args.device)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        # y = self.selective_scan(x, delta, A, B, C, D)
        y = self.selective_scan_parallel(x.permute(0,2,1), 
                                         delta.permute(0,2,1), A, B.permute(0,2,1), C.permute(0,2,1), D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D): # A, D 是初始化来的，delta, b, c是投出来的，x是输入
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=self.args.device)

        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)

        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D
    
        return y

    def selective_scan_parallel(self,us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
        """
        # B: batch_size, D: dim, N: state dim, L: seqlen
        us: B, D, L 
        dts: B, D, L
        As: D, N
        Bs: B, N, L
        Cs: B, N, L
        Ds: D
        delta_bias: D
        # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
        """
        def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
            """
            partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
            => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
            => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
            => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
            y_i = C_i*h_i + D*u_i
            """
            """
            us, dts: (L, B, D) # L is chunk_size
            As: (D, N)
            Bs, Cs: (L, B, N)
            Ds: (D)
            hprefix: (B, D, N)
            """
            ts = dts.cumsum(dim=0)
            Ats = torch.einsum("dn,lbd->lbdn", As, ts).exp()
            # scale = Ats[-1].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts * us
            dtBus = torch.einsum("lbd,lbn->lbdn", duts, Bs)
            hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0) 
            hs = hs_tmp + Ats * hprefix.unsqueeze(0)
            ys = torch.einsum("lbn,lbdn->lbd", Cs, hs) 
            return ys, hs
    
        dtype = torch.float32
        inp_dtype = us.dtype
        has_D = Ds is not None
        if chunksize < 1:
            chunksize = Bs.shape[-1]
    
        dts = dts.to(dtype)
        if delta_bias is not None:
            dts = dts + delta_bias.view(1, -1, 1).to(dtype)
        if delta_softplus:
            dts = F.softplus(dts)
        
        B, N, L = Bs.shape
        us = us.view(B, -1, L).permute(2, 0, 1).to(dtype)
        dts = dts.view(B, -1, L).permute(2, 0, 1).to(dtype)
        As = As.view(-1, N).to(dtype)
        Bs = Bs.permute(2, 0, 1).to(dtype)
        Cs = Cs.permute(2, 0, 1).to(dtype)
        Ds = Ds.view(-1).to(dtype) if has_D else None
        D = As.shape[0]
        
        oys = []
        hprefix = us.new_zeros((B, D, N), dtype=dtype)
        for i in range(0, L, chunksize):
            ys, hs = selective_scan_chunk(
                us[i:i + chunksize], dts[i:i + chunksize], 
                As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix, 
            )
            oys.append(ys)
            hprefix = hs[-1]
    
        oys = torch.cat(oys, dim=0)
        if has_D:
            oys = oys + Ds * us
        oys = oys.permute(1, 2, 0).view(B, -1, L)
    
        # return oys, hprefix.view(B, D, N)
        return oys.to(inp_dtype) if not return_last_state else (oys.to(inp_dtype), hprefix.view(B, D, N).float())

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0,
    norm_layer = RMSNorm, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, 
                              padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = self.conv(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x + shortcut

def tester():
    b = 1
    l = 256
    d = 8
    d_in = 16
    n = 16

    u = torch.randn(b, l, d_in).to("mps")
    delta = torch.randn(b, l, d_in).to("mps")
    A = torch.randn(d_in, n).to("mps")
    B = torch.randn(b, l, n).to("mps")
    C = torch.randn(b, l, n).to("mps")
    D = torch.randn(d_in).to("mps")

    us = u.permute(0, 2, 1)
    dts = delta.permute(0, 2, 1)
    As = A
    Bs = B.permute(0,2,1)
    Cs = C.permute(0,2,1)
    Ds = D
    
    mamba = MambaBlock(args = ModelArgs())
    start_time = time.time()
    sequential_out = mamba.selective_scan(u, delta, A, B, C, D)
    sequential_time = time.time() - start_time

    start_time = time.time()
    parallel_out = mamba.selective_scan_parallel(us, dts, As, Bs, Cs, Ds)

    parallel_time = time.time() - start_time

    print(f"It took {sequential_time}s to perform sequential computation. And it took {parallel_time}s to compute perform parallel computation.")
    print(f"The ratio of S/P is {sequential_time/parallel_time: .2f}")

tester()
