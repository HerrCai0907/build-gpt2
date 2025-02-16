import gpt2_model as my_gpt2_model
import torch
import tiktoken
from typing import List
from torch.nn import functional as F
from select_device import device


def run_gpt2(
    model: my_gpt2_model.GPT,
    input: str,
    expected_size: int,
    num_return_sequences: int = 1,
) -> List[str]:
    model.to(device=device)
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(input)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens
    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    while x.size(1) < expected_size:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)
    # print the generated text
    return [
        enc.decode(x[i, :expected_size].tolist()) for i in range(num_return_sequences)
    ]
