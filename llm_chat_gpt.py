from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTOSSModel import *
from BaseModel import *
import torch

model_id = "openai/gpt-oss-20b"

gpt = GPTOSSModel(model_id)
_ = gpt.load()


# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = gpt.format_prompt(user_input)
    tokenized_prompt = gpt.tokenizer(prompt, return_tensors='pt')
    
    tokenized_response = gpt.model.generate(**tokenized_prompt, max_new_tokens=1000)
    full_response = gpt.tokenizer.decode(tokenized_response[0], skip_special_tokens = True)
    response = gpt.extract_response(full_response)
    gpt.update_chat_history(user_input, response)
    print("AI:", response)

