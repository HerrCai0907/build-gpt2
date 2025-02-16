import train as my_gpt2_train
import gpt2_call as my_gpt2_call


model = my_gpt2_train.train()

print("\n".join(my_gpt2_call.run_gpt2(model, "I speak from certainties", 20, 5)))
