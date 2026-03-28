import torch

class CUDAPrefetcher:
    """
    Prefetches batches to GPU asynchronously to overlap data transfer with computation.
    Works with any PyTorch DataLoader.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        for inputs, targets in self.loader:
            # Move the batch to GPU asynchronously
            with torch.cuda.stream(self.stream):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            # Wait for the async copy to finish before yielding
            torch.cuda.current_stream().wait_stream(self.stream)
            yield inputs, targets

    def __len__(self):
        return len(self.loader)