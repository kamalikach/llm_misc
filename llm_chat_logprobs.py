from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import *
from LlamaModel import *
import argparse
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llama = LlamaModel(model_id)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
_ = llama.load(device)


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
        tokenized_prompt = llama.tokenizer(prompt, return_tensors='pt').to(llama.device)
    
        model_outputs = llama.model.generate(**tokenized_prompt, max_new_tokens=300, 
                temperature=args.temperature, 
                do_sample=True, 
                pad_token_id=llama.tokenizer.eos_token_id, 
                return_dict_in_generate=True,
                output_scores=True)

        tokenized_response = model_outputs.sequences
        full_response = llama.tokenizer.decode(tokenized_response[0], skip_special_tokens = True)
        response = llama.extract_response(full_response)
        llama.update_chat_history(user_input, response)
        print("AI:", response)

        scores = model_outputs.scores  # list of logits for each generated token
        generated_tokens = tokenized_response[0][tokenized_prompt["input_ids"].shape[1]:]  
        log_probs = []
        for i, score in enumerate(scores):
            log_softmax = torch.nn.functional.log_softmax(score, dim=-1)
            token_id = generated_tokens[i]
            log_probs.append(log_softmax[0, token_id].item())

        avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else float("nan")
        print("Log Probabilites:",log_probs)
        print("Average:", avg_log_prob)

if __name__ == "__main__":
    main()
