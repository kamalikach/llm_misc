from transformers import AutoModelForCausalLM, AutoTokenizer
from LlamaModel import *
from BaseModel import *
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Example for LLaMA 3.1 8B model

llama = LlamaModel(model_id)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
_ = llama.load(device)


# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = llama.format_prompt(user_input)
    tokenized_prompt = llama.tokenizer(prompt, return_tensors='pt').to(llama.device)
    
    tokenized_response = llama.model.generate(**tokenized_prompt, max_new_tokens=300, top_p=0.95, do_sample=True)
    full_response = llama.tokenizer.decode(tokenized_response[0], skip_special_tokens = True)
    response = llama.extract_response(full_response)
    llama.update_chat_history(user_input, response)
    print("Llama:", response)

