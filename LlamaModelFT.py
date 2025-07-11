from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import BaseModel
from peft import PeftModel

class LlamaModelFT(BaseModel):
    def load(self, directory, device):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        ft_model = PeftModel.from_pretrained(base_model, directory)
        ft_model = ft_model.merge_and_unload()

        self.model = ft_model
        self.device = device
        self.tokenizer = tokenizer
        self.chat_history = []
        self.model.config.pad_token_id = self.model.config.eos_token_id

        return tokenizer, ft_model

    def format_prompt(self, user_input):
        messages = self.chat_history + [{"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def update_chat_history(self, user_input, response):
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})

    def extract_response(self, full_response):
        # Lowercase to match role labels, can adjust based on your template
        assistant_tag = "assistant"
        # Find the last occurrence of 'assistant:'
        idx = full_response.lower().rfind(assistant_tag)
        if idx == -1:
            # fallback: just return the full text
            return full_response.strip()
        # Extract everything after 'assistant:'
        reply = full_response[idx + len(assistant_tag):].strip()
        # Optionally stop at next role mention if model continues beyond
        # For example, stop if 'user:' appears after reply
        next_user_idx = reply.lower().find("user:")
        if next_user_idx != -1:
            reply = reply[:next_user_idx].strip()
        return reply
