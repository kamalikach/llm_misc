from transformers import AutoModelForCausalLM, AutoTokenizer
import BaseModel

class LlamaModel(BaseModel):
    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model = model
        self.tokenizer = tokenizer
        self.chat_history = ""

        return tokenizer, model

    def format_prompt(self, user_input):
        messages = self.chat_history + [{"role": "user", "content": user_input}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

