from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Example for LLaMA 3.1 8B model

def format_llama3(user_input):
    return f"<|start_header_id|>user<|end_header_id|>\n{user_input.strip()}\n<|start_header_id|>assistant<|end_header_id|>\n"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

# Create chat pipeline
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = format_llama3(user_input)
    response = chat(prompt, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
    
    print(f"LLaMA: {response.replace(prompt, '').strip()}")
