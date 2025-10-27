import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import BaseModel  # assuming you saved your ABC in base_model.py

class GPTOSSModel(BaseModel):
    def __init__(self, model_id: str = "gpt-oss:20b"):
        super().__init__(model_id)
        self.model = None
        self.tokenizer = None
        self.chat_history = []

    def load(self):
        """Load the GPT-OSS model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        return self

    def format_prompt(self, user_input: str) -> str:
        """Format prompt with history (simple version)."""
        conversation = ""
        for turn in self.chat_history:
            conversation += f"User: {turn['user']}\nassistant: {turn['assistant']}\n"
        conversation += f"User: {user_input}\nassistant:"
        return conversation

    def extract_response(self, full_response: str) -> str:
        """Extract only the assistant's reply from the raw text."""
        if "Assistant:" in full_response:
            return full_response.split("assistant:")[-1].strip()
        return full_response.strip()

    def update_chat_history(self, user_input, response):
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})

