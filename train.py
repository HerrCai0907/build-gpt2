import tiktoken
import torch
from select_device import device
import gpt2_model as my_gpt2_model


with open("input/tiny_shakespeare.txt", "r") as f:
    text = f.read()

data = text[:1000]

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)

B, T = 4, 32


buf = torch.tensor(tokens[: B * T + 1], device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = my_gpt2_model.GPT(my_gpt2_model.GPTConfig())
model.to(device=device)

i = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
logits, loss = model(x, y)
while loss.item() > 0.1:
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
    i += 1
