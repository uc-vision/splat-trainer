import torch


def _assign_points_to_clusters(features: torch.Tensor, centroids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Assigns points to the nearest centroid with approximate capacity using weighting."""
    k = centroids.shape[0]
    new_labels = torch.full((features.size(0),), fill_value=-1, dtype=torch.long, device=features.device)
    distances = torch.cdist(features, centroids)
    
    # Calculate weighted distances
    weighted_distances = distances * weights
    
    # Assign each point to the cluster with the minimum weighted distance
    new_labels = torch.argmin(weighted_distances, dim=1)
    
    return new_labels


def _update_centroids(features: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Updates centroids based on the current cluster assignments."""
    centroids = torch.zeros((k, features.shape[1]), dtype=features.dtype, device=features.device)

    # Start of Selection
    counts = torch.bincount(labels, minlength=k).unsqueeze(-1).float()
    sums = torch.zeros_like(centroids).scatter_add(0, labels.unsqueeze(-1).expand(-1, features.shape[1]), features)
    return sums / counts.clamp(min=1)


def kmeans_constrained(features: torch.Tensor, target_n: int, max_iter:int=100, alpha:float=0.2):
    # Initialize centroids randomly
    k = features.shape[0] // target_n


    centroids = features[torch.randperm(features.size(0))[:k]]
    labels = torch.full((features.size(0),), fill_value=-1, dtype=torch.long, device=features.device)
    
    weights = torch.ones(k, dtype=torch.float, device=features.device) # Initialize weights to 1

    for _ in range(max_iter):  # Maximum iterations
        
        new_labels = _assign_points_to_clusters(features, centroids, weights)
        labels = new_labels
        
        # Calculate cluster counts from previous labels
        cluster_counts = torch.bincount(labels[labels != -1], minlength=k).float()
        
        # Calculate target weights based on cluster sizes
        target_weights = target_n / (cluster_counts + 1e-6)  # Adding a small value to avoid division by zero
        
        # Update weights using a moving average
        weights = (1 - alpha) * weights + alpha * target_weights

        centroids = _update_centroids(features, labels, k)


    return labels