import torch

def evaluate_failure(model, clean_images, corrupted_images):
    """
    Compares confidence drop between clean and corrupted images
    """
    clean_conf = []
    corrupted_conf = []

    for clean_img, corrupt_img in zip(clean_images, corrupted_images):
        clean_prob = model.predict(clean_img)
        corrupt_prob = model.predict(corrupt_img)

        clean_conf.append(torch.max(clean_prob).item())
        corrupted_conf.append(torch.max(corrupt_prob).item())

    confidence_drop = [
        c - f for c, f in zip(clean_conf, corrupted_conf)
    ]

    avg_drop = sum(confidence_drop) / len(confidence_drop)

    return avg_drop, confidence_drop
