# toy-transformer
A toy transformer project.

## Forward Computation
What am I going to build?
```py
x = x + self.attn(self.layer_norm_1(x))
x = x + self.mlp(self.layer_norm_2(x))
```
+ Layer normalization for input
+ residual conecction: both attn and mlp

### Self Attention

Inherent from nn.Module
#### `Q`, `K`, `V` Projection

Actually, this is a Linear Project. The input dimension is identical with the input token and output dimensions are $N_{heads} * 3$
```python
q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
```
### LayerNorm
+ Do the normalization $\sum^{d}_{i=1} (x_i - \mu_i) / (\sigma + \epsilon)$ and then apply the linear projection
$y = w x + b$
```py
from torch.nn import functional as F
F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```
where
+ `input.shape = (B, T, C)`. `B` is the batch size, `T` is the number of tokens, `C` is embedding dimension.
+ `self.weight.shape` is the size of this layer, namely, the dimension of embedding
> **LayerNorm**: To normalize on all the token vectors respectively. For every single tokens (B, T, C) = (1, 1, C), collapse the `C` dimension, so we have the center vector is (B, T, C) = (1, T, 1) for this batch. <br>
> 
> **BatchNorm**: To normalize the batch input respectively on every token, the center vector is `(B, T, C) = (1, 1, C)` for this batch. For every feature, collapse the `B, T` dimensions to get the average.
+ Functions:
  + Init(): initialize $W_K, W_Q, W_V$ and hyper-parameters.
  + Forward(): Matrix Products, Q, K, V and then Softmax

### Word Segment
+ Byte Level BPE
+ BPE
+ WordPiece
+ Unigram LM


### Position Encoding
Idea of position embedding: 

Position embeddings should have the following properties
- PE of every position should be unique
- Independent from training
- As $|i - j|$ grows, $P(i) P(j)$ decays

[SuJianlin's Blog](https://zhuanlan.zhihu.com/p/359500899).For any position $i, j$, position $P(k)$, satisfies $P(i) P(j) = P(i - j)$

- `SIN` is one of the solutions, but has the disadvantage of "[pollution](https://zhuanlan.zhihu.com/p/1963547389718692709)", because the embedding is `added` to the token.
- [`RoPE`](https://zhuanlan.zhihu.com/p/2001987112526947375): Based on `multiplication`.
  - rotate the 2D sub vectors, respectively
  - the rotation matrix is sparse
  - the value of $\theta$ is similar to those in `SIN` encoding.

```py
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

> Position encoding allocates a Base-$\beta$-system for position $i$ of the token vector.

> One of the bottlenecks in long context: The extrapolation of position encoding in test-time. The position embedding out of the range are not trained.

### Buildup Models in Blocks
- Single Layer Transformer
```py
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```
- GPT2

## Training

###  Read Configuration

### Initialize Runtime Environment

### Prepare Data Interface

### Load model
- Access files in memory-mapping method, the file on the disk is seen as a numpy array.
- If using GPUs:
  - `pin_memory()` method should be applied to the numpy array so that the memory pages won't be swapped to the disk.
  - should use `non-blocking` to move data to GPUs asynchronously without blocking current process.

The most naive data loader:
```py
def get_batch(split):
  # which subset
  if (split == training):
    # data won't be loaded into DDR, still on the disk
    data = np.memmap("path_to_file/train.bin"， dtype=np.uint16, mode='r');
  else:
    data = np.memmap("path_to_file/train.bin"， dtype=np.uint16, mode='r');

  # cpu or GPU
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
       # load data to RAM or DDR
        x, y = x.to(device), y.to(device)
    return x, y
```
### Setup Optimizer and Tools

### Training Loop
- A naive training loop
```py
while True:
    X, Y = get_batch('train')

    logits, loss = model(X, Y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    iter_num += 1
```
**Training techniques** 
- Gradient Accumulation: `loss = (loss_1 + loss_2 + ... + loss_N) / N`
- Mixed Precision：`scaler.scale(loss).backward()`
- DDP (distributed data parallel): Only synchronize the gradient on the last micro-step: `model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)`
- eval / save / log
- gradient clip

### Save, Quit, Cleanup
