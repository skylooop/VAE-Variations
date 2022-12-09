import torch
import torch.nn as nn
import pyrallis
from dataclasses import dataclass, field
from ..utils.gumbel import gumbel_softmax, sample_gumble
from network import VAE_gumbel



@dataclass
class GumbelSoftMax_cfg:
    batch_size: int = field(default=100)
    epochs: int = field(default=10)
    latent_dim: int = field(default=30)
    categorical_dim: int = field(default=10)
    thau: float = field(default=1.0)


    def __post_init__(self):
        self.device = torch.device("cuda")
        print(f"using cuda: {self.device}")
    
def main():
    print(gumbel_softmax(torch.rand((1, 3))))

if __name__ == "__main__":
    main()
