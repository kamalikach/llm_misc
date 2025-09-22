import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import BaseModel  # assuming you saved your ABC in base_model.py

class GPTOSSModel(BaseModel):
    def __init__(self, model_id: str = "gpt-oss:20b"):
        super().__init__(model_id)
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self, device: str = None):
        """Load the GPT-OSS model and tokenizer."""
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None
        )
        return self

    def format_prompt(self, user_input: str) -> str:
        """Format prompt with history (simple version)."""
        conversation = ""
        for turn in self.chat_history:
            conversation += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        conversation += f"User: {user_input}\nAssistant:"
        return conversation

    def extract_response(self, full_response: str) -> str:
        """Extract only the assistant's reply from the raw text."""
        if "Assistant:" in full_response:
            return full_response.split("Assistant:")[-1].strip()
        return full_response.strip()
