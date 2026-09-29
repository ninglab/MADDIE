from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
from ..data.route import Route

class Tool(ABC):
    def __init__(self, **kwargs):
        self.setup_config = kwargs

    def setup_tool(self):
        """Configures the tool using the dictionary of keyword arguments stored in self.setup_config."""
        pass

    @abstractmethod
    def documentation(self) -> str:
        """Instructions to agent for the tool."""
        return NotImplemented

    @abstractmethod
    def __call__(self, **params) -> Tuple[Union[Route, List[Route], None], str]:
        """Executes the tool using the dictionary of parameters specified, returning a (mergable) raw data result and a short string status, respectively."""
        pass
        
