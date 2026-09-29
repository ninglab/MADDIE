from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

class BaseMemory:
    def __init__(
        self,
        path_to_data: str,
    ):
        self.path_to_data = path_to_data
    
    @abstractmethod
    def query(self):
        """Query the memory"""
    
    def store(self):
        """Store new information in the memory"""