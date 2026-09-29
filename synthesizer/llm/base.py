from abc import abstractmethod, ABC

class LLM(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_single_response(self, user, system=None) -> str:
        pass