import torch
from skimage import measure

def remove_small_frontiers(frontier_map, min_size=1):
    device = frontier_map.device
    frontier_np = frontier_map.cpu().numpy().astype(bool)

    labeled = measure.label(frontier_np, connectivity=2)

    label_sizes = {}
    for label in range(1, labeled.max() + 1):
        label_sizes[label] = (labeled == label).sum()

    output_np = torch.zeros_like(frontier_map)
    for label in label_sizes:
        if label_sizes[label] >= min_size:
            output_np[labeled == label] = 1

    return output_np.to(device)