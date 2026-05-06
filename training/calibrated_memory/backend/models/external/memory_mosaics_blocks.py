
# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
#
# This file is derived from nanoGPT.
# See LICENSE-nanoGPT.md for the original license.


"""
Full definition of a Memory Mosaic Language Model, all of it in this single file.
This is intentionally kept as close as possible to the original gpt_model.py.
"""



import math
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

try:
    from torch.nn.attention import sdpa_kernel as torch_sdpa_kernel
except (ImportError, AttributeError):
    torch_sdpa_kernel = None
try:
    from torch.backends.cuda import sdp_kernel as torch_cuda_sdp_kernel
except (ImportError, AttributeError):
    torch_cuda_sdp_kernel = None

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LeakyAvg(nn.Module):
    def __init__(self, block_size, n_head):
        super().__init__()
        # The dynamic implementation no longer stores a static coefficient buffer, so
        # the former block_size parameter now simply controls initialization scale.
        self.block_size = block_size
        self.exp_scaling = 10
        self.leaky_key_beta = nn.Parameter(
            torch.linspace(0.5, 5, n_head).view(1, n_head, 1, 1) / self.exp_scaling
        )

    def get_decay_rates(self) -> torch.Tensor:
        return torch.exp(-self.leaky_key_beta.abs() * self.exp_scaling)

    def forward(
        self,
        k: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_recurrent_mode: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_recurrent_mode:
            return self.forward_recurrent(k, state)
        return self.forward_parallel(k)

    def forward_parallel(self, k: torch.Tensor) -> torch.Tensor:
        B, nh, T, _ = k.size()
        leaky_key_beta = self.leaky_key_beta.abs() * self.exp_scaling
        t_idx = torch.arange(T, device=k.device)
        diff = t_idx.view(1, 1, T, 1) - t_idx.view(1, 1, 1, T)
        diff = diff.to(dtype=k.dtype)
        decay_logits = diff * (-leaky_key_beta)
        neg_inf = torch.finfo(k.dtype).min
        decay_logits = decay_logits.masked_fill(diff < 0, neg_inf)
        coef = torch.exp(decay_logits)
        coef = torch.tril(coef)
        coef = coef.expand(B, -1, -1, -1)
        return coef @ k

    def forward_recurrent(
        self,
        k: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = self.get_decay_rates().to(dtype=k.dtype, device=k.device)
        decay = decay.view(1, -1, 1, 1)
        if state is None:
            state = torch.zeros(
                k.size(0), k.size(1), 1, k.size(-1), device=k.device, dtype=k.dtype
            )
        outputs = []
        curr_state = state
        for t in range(k.size(2)):
            k_t = k[:, :, t:t+1, :]
            curr_state = k_t + decay * curr_state
            outputs.append(curr_state)
        y = torch.cat(outputs, dim=2)
        return y, curr_state

class KeyFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.leaky_cuda = config.leaky_cuda
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.leaky_avg = LeakyAvg(config.block_size, config.n_head)
        self.exp_scaling = 10
        self.key_scale = nn.Parameter(torch.ones(1, config.n_head, 1, 1) / self.exp_scaling)
        self.key_scale_max = math.log(2**16-1) # fits in fp16.

    def forward(
        self,
        x,
        scale_pow=1,
        state: Optional[torch.Tensor] = None,
        use_recurrent_mode: bool = False,
    ):
        B,T,C = x.size()
        hs = C // self.n_head
        k = self.W_k(x).transpose(1,2).view(B, self.n_head, hs, T).transpose(2,3)
        if use_recurrent_mode:
            k, new_state = self.leaky_avg(k, state=state, use_recurrent_mode=True)
        else:
            k = self.leaky_avg(k)
            new_state = None
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-10)
        k = k * (scale_pow * self.exp_scaling * self.key_scale).clamp(max=self.key_scale_max).exp()
        if use_recurrent_mode:
            return k, new_state
        return k

class ValFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        v_shift = 1 # access to x_T+1
        self.shift_fn = lambda x: F.pad(x, (-v_shift, v_shift))
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.coef = nn.Parameter(torch.rand(1, config.n_head, 1, 1))
        self.exp_scaling = 10
        val_scale_init = -.5
        self.val_scale  = nn.Parameter(torch.ones(1, config.n_head, 1, 1) * val_scale_init / self.exp_scaling)

    def forward(
        self,
        x,
        state: Optional[torch.Tensor] = None,
        use_recurrent_mode: bool = False,
    ):
        B,T,C = x.size()
        hs = C // self.n_head
        v_raw = self.W_v(x).transpose(1,2).view(B, self.n_head, hs, T)
        if use_recurrent_mode:
            return self.forward_recurrent(v_raw, state)
        v = (1-self.coef) * self.shift_fn(v_raw) + self.coef * v_raw
        v = v.transpose(2,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-10)
        v = v * (self.exp_scaling * self.val_scale).exp()
        return v

    def forward_recurrent(
        self,
        v_raw: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        outputs = []
        pending = state
        scale = (self.exp_scaling * self.val_scale).exp()
        for t in range(v_raw.size(-1)):
            this_raw = v_raw[..., t:t+1]
            if pending is not None:
                val_prev = (1 - self.coef) * this_raw + self.coef * pending
                val_prev = val_prev.transpose(2, 3)
                val_prev = val_prev / (val_prev.norm(dim=-1, keepdim=True) + 1e-10)
                val_prev = val_prev * scale
                outputs.append(val_prev)
            pending = this_raw
        if outputs:
            values = torch.cat(outputs, dim=2)
        else:
            values = None
        return values, pending

    def finalize_state(self, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if state is None:
            return None
        scale = (self.exp_scaling * self.val_scale).exp()
        final = self.coef * state
        final = final.transpose(2, 3)
        final = final / (final.norm(dim=-1, keepdim=True) + 1e-10)
        final = final * scale
        return final

class ContextMem(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.k_featurizer = KeyFeatureExtractor(config)
        self.v_featurizer = ValFeatureExtractor(config)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.attn_dropout = nn.Dropout(config.dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def _sdp_kernel_context(self, tensor: torch.Tensor):
        if self.flash and tensor.is_cuda and torch_sdpa_kernel is not None:
            try:
                return torch_sdpa_kernel(
                    enable_flash=True,
                    enable_mem_efficient=True,
                    enable_math=True,
                )
            except TypeError:
                try:
                    return torch_sdpa_kernel(
                        flash=True,
                        mem_efficient=True,
                        math=True,
                    )
                except TypeError:
                    pass
        if self.flash and tensor.is_cuda and torch_cuda_sdp_kernel is not None:
            return torch_cuda_sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=True,
            )
        return nullcontext()

    def forward(
        self,
        x,
        layer_past: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
        use_recurrent_mode: bool = False,
    ):
        B, T, C = x.size()
        leaky_state: Optional[torch.Tensor]
        cache_k: Optional[torch.Tensor]
        cache_v: Optional[torch.Tensor]
        pending_keys: Optional[torch.Tensor]
        pending_val_state: Optional[torch.Tensor]
        if layer_past is not None:
            if len(layer_past) < 3:
                raise ValueError(
                    "layer_past must contain at least (leaky_state, cache_k, cache_v) entries"
                )
            leaky_state = layer_past[0]
            cache_k = layer_past[1]
            cache_v = layer_past[2]
            pending_keys = layer_past[3] if len(layer_past) > 3 else None
            pending_val_state = layer_past[4] if len(layer_past) > 4 else None
        else:
            leaky_state = None
            cache_k = None
            cache_v = None
            pending_keys = None
            pending_val_state = None

        if use_recurrent_mode:
            if T != 1:
                raise ValueError("Recurrent mode currently supports one token per call")
            k, new_leaky_state = self.k_featurizer(x, state=leaky_state, use_recurrent_mode=True)
            v_ready, new_pending_val_state = self.v_featurizer(
                x, state=pending_val_state, use_recurrent_mode=True
            )
            if pending_keys is None:
                pending_keys = k
            else:
                pending_keys = torch.cat([pending_keys, k], dim=2).contiguous()
            ready_len = 0 if v_ready is None else v_ready.size(2)
            if ready_len:
                if pending_keys is None or pending_keys.size(2) < ready_len:
                    raise RuntimeError("Value/state mismatch when finalizing Memory Mosaic cache")
                ready_keys = pending_keys[:, :, :ready_len, :]
                pending_keys = pending_keys[:, :, ready_len:, :]
                cache_k = ready_keys if cache_k is None else torch.cat([cache_k, ready_keys], dim=2).contiguous()
                cache_v = v_ready if cache_v is None else torch.cat([cache_v, v_ready], dim=2).contiguous()
            if pending_keys is not None and pending_keys.size(2) == 0:
                pending_keys = None
            history_k = cache_k
            history_v = cache_v
            if history_k is None or history_v is None or history_k.size(2) == 0:
                nh = k.size(1)
                hs = k.size(-1)
                y = torch.zeros(B, nh, T, hs, device=x.device, dtype=x.dtype)
            else:
                cache_ctx = self._sdp_kernel_context(history_k)
                with cache_ctx:
                    if self.flash:
                        y = torch.nn.functional.scaled_dot_product_attention(
                            k,
                            history_k,
                            history_v,
                            attn_mask=None,
                            dropout_p=self.dropout if self.training else 0,
                            is_causal=False,
                        )
                    else:
                        att = k @ history_k.transpose(-2, -1)
                        att = F.softmax(att, dim=-1)
                        att = self.attn_dropout(att)
                        y = att @ history_v
            new_layer_past = (new_leaky_state, cache_k, cache_v, pending_keys, new_pending_val_state)
        else:
            k = self.k_featurizer(x)
            v = self.v_featurizer(x)
            y = torch.zeros_like(v)
            if T > 1:
                q_ctx = self._sdp_kernel_context(k[:, :, 1:, :])
                with q_ctx:
                    if self.flash:
                        y[:, :, 1:] = torch.nn.functional.scaled_dot_product_attention(
                            k[:, :, 1:, :],
                            k[:, :, :-1, :],
                            v[:, :, :-1, :],
                            attn_mask=None,
                            dropout_p=self.dropout if self.training else 0,
                            is_causal=True,
                        )
                    else:
                        att = k[:, :, 1:, :] @ k[:, :, :-1, :].transpose(-2, -1)
                        mask = torch.tril(
                            torch.ones(T-1, T-1, device=x.device, dtype=torch.bool)
                        ).unsqueeze(0).unsqueeze(0)
                        att = att.masked_fill(~mask, float('-inf'))
                        att = F.softmax(att, dim=-1)
                        att = self.attn_dropout(att)
                        y[:, :, 1:] = att @ v[:, :, :-1, :]
            new_layer_past = None

        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_dropout(self.c_proj(y))

        if use_recurrent_mode:
            return y, new_layer_past
        return y

class PersistentMem(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, value projections for all heads, but in a batch
        self.k_featurizer = KeyFeatureExtractor(config)
        pmem_dim = config.n_embd // config.n_head
        self.P_k = nn.Parameter(torch.zeros(config.pmem_count, 1, config.n_head, config.pmem_size, pmem_dim))
        self.P_v = nn.Parameter(torch.zeros(config.pmem_count, 1, config.n_head, config.pmem_size, pmem_dim))
        self.exp_scaling = 10
        out_scale_init = -.5
        self.out_scale  = nn.Parameter(torch.ones(1, config.n_head, 1, 1) * out_scale_init / self.exp_scaling)
        torch.nn.init.normal_(self.P_k, mean=0.0, std=1 / math.sqrt(pmem_dim))
        torch.nn.init.normal_(self.P_v, mean=0.0, std=1 / math.sqrt(pmem_dim))

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.pmem_count = config.pmem_count
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if not self.flash:
            self.attn_dropout = nn.Dropout(config.ic_dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_featurizer(x, scale_pow=2) # 2 because P_k does not have scale

        if self.flash:
            y = 0
            for i in range(self.pmem_count):
                y = y + F.scaled_dot_product_attention(
                    k,
                    self.P_k[i],
                    self.P_v[i],
                    scale=1,
                    dropout_p=self.dropout if self.training else 0,
                )
        else:
            # manual implementation of attention
            for i in range(self.pmem_count):
                att = k @ (self.P_k[i].transpose(-2, -1))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y += att @ self.P_v[i] # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y / self.pmem_count
        y = y * (self.exp_scaling * self.out_scale).exp()
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = ContextMem(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = PersistentMem(config)

    def forward(self, x, layer_past: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None, use_recurrent_mode: bool = False):
        if use_recurrent_mode:
            attn_out, new_layer_past = self.attn(
                self.ln_1(x), layer_past=layer_past, use_recurrent_mode=True
            )
        else:
            attn_out = self.attn(self.ln_1(x))
            new_layer_past = None
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if use_recurrent_mode:
            return x, new_layer_past
        return x

@dataclass
class MemoryMosaicConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    pmem_size: int = 2688
    pmem_count: int = 1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    leaky_cuda: bool = False # True: use LeakyAverageCuda, False: use LeakyAvg

class MemoryMosaic(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.mosaic = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.mosaic.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.mosaic.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.mosaic.drop(tok_emb)
        for block in self.mosaic.h:
            x = block(x)
        x = self.mosaic.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.mosaic.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        leave_out = lambda x : x.dim() < 2 or x.shape[-2:] == (1, 1)
        decay_params = [p for n, p in param_dict.items() if not leave_out(p)]
        nodecay_params = [p for n, p in param_dict.items() if leave_out(p)]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
