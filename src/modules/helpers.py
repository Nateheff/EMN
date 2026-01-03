import torch

def kWTA(input, k):

    # Get the k-th largest value
    # topk returns values and indices, we need the k-th value
    # torch.topk returns values in descending order, so the k-th value is at index k-1
    kth_value = torch.topk(input, k, dim=-1)
    thresh = kth_value.values[:, -1].unsqueeze(-1)

    # Create a mask where values greater than or equal to the k-th value are True
    mask = (input >= thresh).to(input.dtype)

    # Apply the mask to the original tensor
    return input * mask
