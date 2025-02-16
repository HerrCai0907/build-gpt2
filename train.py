import tiktoken
import torch
from select_device import device
import gpt2_model as my_gpt2_model


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open("input/tiny_shakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].to(
            device=device
        )
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


train_loader = DataLoaderLite(B=4, T=32)


model = my_gpt2_model.GPT(my_gpt2_model.GPTConfig())
model.to(device=device)

i = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
while True:
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
    i += 1
    if loss.item() < 0.1:
        break
