from transformers import AutoModelForCausalLM, AutoTokenizer
from LlamaModelFT import *
from BaseModel import *
import torch

model_id = "meta-llama/Meta-Llama-3-8b-Instruct"
ckpt_dir = "/data/pengrun/dia_models/propInfer_series/Llama3-CC/data_30_6500_rs_300_Llama-3-8b_input"

llama = LlamaModelFT(model_id)
_ = llama.load(ckpt_dir)


# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = llama.format_prompt(user_input)
    tokenized_prompt = llama.tokenizer(prompt, return_tensors='pt')
    
    tokenized_response = llama.model.generate(**tokenized_prompt, max_new_tokens=300, top_p=0.95, do_sample=True, pad_token_id=llama.tokenizer.eos_token_id)
    full_response = llama.tokenizer.decode(tokenized_response[0], skip_special_tokens = True)
    response = llama.extract_response(full_response)
    llama.update_chat_history(user_input, response)
    print("AI:", response)

