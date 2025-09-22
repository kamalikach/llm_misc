from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import *
from LlamaModel import *
import argparse
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llama = LlamaModel(model_id)
_ = llama.load()


# Chat loop
def main():
    parser = argparse.ArgumentParser(description="LLM chat with different temperatures")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")

    args = parser.parse_args()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        prompt = llama.format_prompt(user_input)
        tokenized_prompt = llama.tokenizer(prompt, return_tensors='pt').to(llama.model.device) 
        tokenized_response = llama.model.generate(**tokenized_prompt, max_new_tokens=300, temperature=args.temperature, do_sample=True, pad_token_id=llama.tokenizer.eos_token_id)
        full_response = llama.tokenizer.decode(tokenized_response[0], skip_special_tokens = True)
        response = llama.extract_response(full_response)
        llama.update_chat_history(user_input, response)
        print("AI:", response)

if __name__ == "__main__":
    main()
