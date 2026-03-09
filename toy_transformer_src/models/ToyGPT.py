import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# how to specify the types?
class LayerNorm(nn.Module):
    def __init__(self,
                 ndim: int,
                 bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

@dataclass
class ToyGPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024 # max length of a sequence
    n_layer: int = 12
    n_embed: int = 768 # users don't need to specify the dimension of each head
    n_head: int = 12
    bias: bool = False # a vector with length of block_size
    dropout: float = 0.1 # rate of dropout, randomly set some elements to zero during training

class ToySelfAttn(nn.Module):
    def __init__(self, config:ToyGPTConfig):
        '''
        Define some components here
        '''
        super().__init__()
        self.config = config
        # 3 vectors (K, Q, V), each of size n_embed
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        # assert the shape of x
        assert x.dim() == 3, "Input tensor must be 3-dimensional"
        '''
        B: Batch size
        T: Current Sequence length, +1 for each new token
        C: embedded dimensionality (self.config.n_embed)
        '''
        B, T, C = x.shape
        # To map the features to k, q, v, and split on the feature dimension
        q, k, v = self.c_attn(x).split(self.config.n_embed, dim=-1)
        # reshape q, k, v to (B, nh, T, hs)
        q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, nh, T, hs)
        # compute attention scores
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.config.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v #(B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, nh, hs) -> (B, T, C)
        #output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y

class MLP(nn.Module):
    def __init__(self, config:ToyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config:ToyGPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = ToySelfAttn(config)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ToyGPT(nn.Module):

    def __init__(self, config:ToyGPTConfig):
        '''
        Initialize the layers
        '''
        assert config.n_embed > 0, "Embedding dimension must be positive"
        assert config.n_head > 0, "Number of heads must be positive"
        assert config.n_embed % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        assert config.vocab_size > 0, "Vocabulary size must be positive"
        assert config.block_size > 0, "Block size must be positive"
        assert config.n_layer > 0, "Number of layers must be positive"
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed), # token embedding table
            'wpe': nn.Embedding(config.block_size, config.n_embed), # position embedding table
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # stack of blocks
            'ln_f': LayerNorm(config.n_embed, bias=config.bias) # final layer norm
        })
        self.lm_head: nn.Linear(config.n_embed, config.vocab_size, bias=False) # language modeling head
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    '''
    To init weights from a normal distribution, and the bias to zero
    '''
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        '''
        To compute logits and loss (if targets are given)
        '''
        device = idx.device
        '''
        b: batch size
        t: sequence length
        '''
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        # GPT-2 SIN encoding
        tok_emb = self.transformer.wte(idx) + self.transformer.wpe(pos) # (b, t, n_embed)
        pos_emb = self.transformer.drop(tok_emb)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # (b, t, n_embed)
        if target is not None:
            '''
            If targets are givent to compute loss
            '''
            logits = self.transformer.lm_head(x) # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else
            '''
            Inference mode, no need to compute loss
            only forward the lm_head on the very last position
            '''
            logits = self.lm_head(x[:, -1, :]) # (b, vocab_size)
            loss = None
        return logits, loss


    def crop_block_size(self, block_size):
        '''
        To use smaller block size during inference
        '''
        
        # use a smaller block_size
        assert block_size <= self.config.block_size, "New block size must be less than or equal to original block size"
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'mask'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    '''
    Loading pretrained weights from HuggingFace/Transformers
    please pay attention to the "state_dict"
    TODO: print the state_dict
    '''
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        '''
        Load a pretrained model
        '''
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = ToyGPTConfig(**config_args)
        model = ToyGPT(config)
        '''
        parameters (weights and biases) are always stored in the state_dict
        state_dict is really important, for saving, loading, and transferring weights between models
        '''
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


    def config_optimizers(self, weight_decay, learning_rate, betas, device_type):
        '''
        TODO: print the param_dict
        1.weight decay:
        - a technique used to prevent a machine learning model from becoming too "complex" or "extreme."
        - a form of regularization that keeps the model's weights small.
        - Why 2D-Rule? Because the weights encodes the transform and relationships of the tokens,
            but the bias are just "shifters"
        2. Fused: Software Optimization for GPUs, increases training speed
        
        '''
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
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
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        - On A100 GPU, how much of the FLOPs are being used? in percentage.
        """
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
        Inference / autoregressive decoding method.
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # use self(), not forward(), really a gotcha
            # see the __call__ method in nn Module
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the multinomial distribution
            # equivalent of sampling for num_sample times
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
