import torch


@torch.no_grad()
def metrics(
    image_embedding: torch.Tensor, text_embedding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    similarity_matrix = torch.matmul(image_embedding, text_embedding.T) / temperature
    y = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    img2text_match_idx = similarity_matrix.argmax(dim=1)
    text2img_match_idx = similarity_matrix.argmax(dim=0)

    img_acc = (img2text_match_idx == y).float().mean()
    text_acc = (text2img_match_idx == y).float().mean()

    return img_acc, text_acc
