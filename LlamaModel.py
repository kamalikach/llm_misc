from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import BaseModel
import torch

class LlamaModel(BaseModel):
    def load(self, attn_implementation="sdpa"):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation
        )
        self.model = model
        self.tokenizer = tokenizer
        self.chat_history = []
        self.model.config.pad_token_id = self.model.config.eos_token_id

        return tokenizer, model

    def format_prompt(self, user_input):
        messages = self.chat_history + [{"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def update_chat_history(self, user_input, response):
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})

    def extract_response(self, full_response: str) -> str:
        """Extract only the assistant's reply from the raw text."""
        if "assistant" in full_response:
            return full_response.split("assistant")[-1].strip()
        else:
            return full_response.strip()

