import torch
from torchvision.models.optical_flow import raft_small
from .base import BaseModel


class RAFTFlow(BaseModel):
    def load(self):
        self.model = raft_small(pretrained=True)
        self.model.eval()
    
    def inference(self, frame1, frame2):
        with torch.no_grad():
            flow = self.model(frame1, frame2)
        return flow
