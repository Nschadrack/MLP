import torch
from torch.utils.data import Dataset


class XORDataset(Dataset):
    """
    This class serves as dataset for XOR samples with corresponding labels
    generated randomly with a seed
    """
    def __init__(self, n_samples: int) -> None:
        """
        params: 
        n_samples: number of samples to be generated, this must be a positive integer
        """
        super().__init__()

        self.n_samples = n_samples
        self.X = torch.randint(low=0, high=2, size=(n_samples,n_samples), dtype=torch.float32)  # generating the first input
        # self.X_2 = torch.randint(low=0, high=1, size=(n_samples,), dtype=torch.float32)  # generating the second input


    def __len__(self):
        """
        Once len method is applied to XORDataset object, this method will be called by returning the total
        number of samples for dataset
        """
        return self.n_samples 

    def __getitem__(self, index):
        """
        This returns the individual sample at a given index
        """
        first_input = self.X[index].item()
        second_input = self.X[index].item()
        label = 1 if first_input != second_input else 0
        return (first_input, second_input), label.to(dtype=torch.float32)
