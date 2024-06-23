import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# batch_size = 64  # how many independent sequences will we process simultaneously
# block_size = 256  # what is the maximum context length for predict
# max_iters = 5000
# eval_intervals = 500
# learning_rate = 3e-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

debug = True
debug_print = print if debug else lambda *args, **kwargs: None


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear Projection to 4 * n_embed dimensions
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # Gaussian Error Linear Unit, a slightly smoother version of ReLU
        self.act = nn.GELU(approximate="tanh")
        # Linear Projection back to n_embd dimensions
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Number of attention heads must be a factor of n_embd
        # Because we split the n_embd dimensional input into n_head parts
        # This allows us to compute attention in parallel and then concatenate
        # the results
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Linear projection matrices for the query, key, and value vectors as a single matrix
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
            ),
        )  # 1, 1 is batch size and n_head. Why 1 head?

    def forward(self, x):
        # Calculate query, key, value hiddem states for all heads in a batch and move head forward
        # to be the the batch
        # nh is "number of heads"
        # hs is "head size"
        # and C is "number of channels" = nh * hs
        # e.g. in GPT-2 (124M), nh=12, hs=64, C=nh*hs=768 channels in the Transformer

        B, T, C = x.size()  # batch size, context size, n_embd

        qkv = self.c_attn(x)
        debug_print(qkv.shape)

        # Split qkv into q, k and v
        # n_embd * 3 -> n_embd, n_embd, n_embd
        q, k, v = qkv.chunk(3, dim=-1)
        assert (
            q.size() == k.size() == v.size()  # == (B, T, C // 3)
        ), "q, k, v size mismatch"

        # Split q, k, v into n_head parts
        # B, T, (q, k, v) -> B, T, n_head, C // n_head
        # We transpose to make the n_head dimension the second dimension
        # B, T, n_head, c (or n_embd) // n_head -> B, n_head, T, C // n_head
        hs = C // self.n_head

        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # Why is this shape required?
        # Because we want to compute attention for each head in parallel

        # q: B, nh, T, hs
        # k: B, nh, T, hs -> k^T: B, nh, hs, T
        # q*k^T = B, nh, T, hs * B, nh, hs, T = B, nh, T, T
        k_T = k.transpose(-2, -1)
        sqrt_d_k = math.sqrt(q.size(-1))
        attn = (q @ k_T) / sqrt_d_k

        # Mask out the upper half of the dot product matrix, excluding the diagonal
        # In other words, we only want to consider the "past" context
        attn = attn.masked_fill(self.bias[:, :, :, :] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Apply the attention to the values
        y = attn @ v  # B, nh, T, T @ B, nh, T, hs = B, nh, T, hs
        # Weighted average of the values with the attention scores

        y = y.transpose(1, 2)  # B, T, nh, hs
        # contiguous() is required because of the transpose because it doesn't change the memory layout
        y = y.contiguous().view(B, T, C)
        # debug_print(y.shape, C.shape)

        # Linear projection back to n_embd dimensions
        y = self.c_proj(y)

        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Weighted sum of input and attention output
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    context_size: int = 256  # block size is just the context size
    # 64 because: [PAD], [SOS], [EOS], [MASK], [UNK] + 60 characters
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    debug: bool = False


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # word token n_embd
                # vocab_size x n_embd
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # word position embedding
                # context_size x n_embd
                wpe=nn.Embedding(config.context_size, config.n_embd),
                # transformer layers
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
            )
        )

        # linear layer for output before softmax
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

        from transformers import GPT2LMHeadModel

        debug_print(f"Loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            # 350M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["context_size"] = 1024

        # Create a new GPT2 model
        config = GPTConfig(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [key for key in sd_keys if not key.endswith(".attn.bias")]
        # debug_print(f"{sd_keys=}")

        # Init a huggingface/transformwers model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        # debug_print(f"{hf_model}")
        sd_hf = hf_model.state_dict()

        # Copy while ensuring all of the parameter are aligned and match in name and shape
        sd_keys_hf = sd_hf.keys()

        # Remove the attn bias as they are not parameters but used as masks (they are buffers)
        sd_keys_hf = [
            key for key in sd_keys_hf if not key.endswith(".attn.masked_bias")
        ]
        sd_keys_hf = [key for key in sd_keys_hf if not key.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # Openai weights use Conv1D instead of Linear, so we need to transpose the weights
        assert len(sd_keys) == len(sd_keys_hf), "Length mismatch"

        for key in sd_keys_hf:
            if any(key.endswith(w) for w in transposed):
                assert sd_hf[key].shape[::-1] == sd[key].shape, "Shape mismatch"
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].T)
            else:
                assert sd_hf[key].shape == sd[key].shape, "Shape mismatch"
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        return model

    def forward(self, idx):
        # x: (batch_size, context_size) or (B, T)
        B, T = idx.size()  # T <= context_size

        assert T <= self.config.context_size, "Context size mismatch"

        # word position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        # word token embedding
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        # (B, T, n_embd) there is broadcasting happening here
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits


if __name__ == "__main__":
    device = "mps"

    config = GPTConfig()
    model = GPT2(config)
    model.to(device)

    # debug_print(model)

    x = torch.randint(0, 65, (64, 256))
    debug_print(f"{x.shape=}")
    x = x.to(device)
    y = model(x)
    debug_print(f"{y.shape=}")

    model = GPT2.from_pretrained("gpt2")
    # debug_print(model)

    model.eval()  # Good practice to set the model to eval mode when not training
    model.to(device)

    max_return_sequences = 5
    max_length = 30

    # Generate text
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, my name is")
    # long because in pytorch long is used for integer values
    tokens = torch.tensor(tokens, dtype=torch.long)  # (T)
    tokens = tokens.unsqueeze(0).repeat(max_return_sequences)  # (B, T)
    # Without repeat, the tensor would be (1, T). We instead repeat it B times
    # in order to sample B times for the same sequence

    tokens.to(device)

    print(tokens.shape)
