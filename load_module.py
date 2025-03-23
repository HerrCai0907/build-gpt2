import torch
import struct

model = torch.load("gpt2.pt", weights_only=False)
print("Model's state_dict:")


for param_tensor in model.state_dict():
    tenser = model.state_dict()[param_tensor]
    print(param_tensor, "\t", tenser.size())
    f = open(f"models/{param_tensor}", "wb")
    flattened_tensor = tenser.view(-1)  # 将张量展平为一维
    for v in flattened_tensor:
        f.write(struct.pack("f", v.item()))
