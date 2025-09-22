from transformers import AutoModelForCausalLM, AutoTokenizer
from BaseModel import BaseModel
from peft import PeftModel
from LlamaModel import *

class LlamaModelFT(LlamaModel):
    def load(self, directory):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        ft_model = PeftModel.from_pretrained(base_model, directory)
        ft_model = ft_model.merge_and_unload()

        self.model = ft_model
        self.tokenizer = tokenizer
        self.chat_history = []
        self.model.config.pad_token_id = self.model.config.eos_token_id

        return tokenizer, ft_model

