import torch


@torch.no_grad()
def top_k_accuracy(
    image_embedding: torch.Tensor, text_embedding: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate top-k accuracy for image-to-text and text-to-image retrieval.

    Args:
        image_embedding: Image embeddings of shape (batch_size, embed_dim)
        text_embedding: Text embeddings of shape (batch_size, embed_dim)
        k: Number of top predictions to consider (default: 5)

    Returns:
        img_acc: Top-k accuracy for image-to-text retrieval
        text_acc: Top-k accuracy for text-to-image retrieval
    """
    similarity_matrix = image_embedding @ text_embedding.T
    y = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)

    # Get top-k indices for image-to-text (each image finds top-k texts)
    img2text_topk_idx = similarity_matrix.topk(k, dim=1).indices  # (batch_size, k)
    # Check if correct index is in top-k for each sample
    img_acc = (img2text_topk_idx == y.unsqueeze(1)).any(dim=1).float().mean()

    # Get top-k indices for text-to-image (each text finds top-k images)
    text2img_topk_idx = similarity_matrix.topk(k, dim=0).indices  # (k, batch_size)
    # Check if correct index is in top-k for each sample
    text_acc = (text2img_topk_idx == y.unsqueeze(0)).any(dim=0).float().mean()

    return img_acc, text_acc
