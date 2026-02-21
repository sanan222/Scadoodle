from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.model = None
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def inference(self, *args, **kwargs):
        pass
