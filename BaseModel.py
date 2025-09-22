from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def format_prompt(self, user_input):
        pass

    @abstractmethod
    def update_chat_history(self, user_input, response):
        pass

    @abstractmethod
    def extract_response(self, full_response):
        pass


