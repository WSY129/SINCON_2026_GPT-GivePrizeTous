"""Validation-ranked averaging; final refits use the last three planned epochs."""
import torch
class TopCheckpoints:
    def __init__(self,k=3):self.k=k;self.items=[]
    def add(self,model,epoch,score=None):
        rank=float(score) if score is not None else -float(epoch)
        self.items.append((rank,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
        self.items=sorted(self.items,key=lambda x:x[0])[:self.k]
    def apply(self,model):
        state={}
        for key,value in self.items[0][2].items():
            state[key]=torch.stack([v[2][key] for v in self.items]).mean(0) if value.is_floating_point() else value
        model.load_state_dict(state)
        return [v[1] for v in self.items]
