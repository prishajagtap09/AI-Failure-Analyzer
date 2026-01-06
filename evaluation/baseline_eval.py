import torch

def evaluate_baseline(model, images):
    """
    Runs model on clean images
    Returns average confidence score
    """
    confidences = []

    for img in images:
        probs = model.predict(img)
        max_confidence = torch.max(probs).item()
        confidences.append(max_confidence)

    avg_confidence = sum(confidences) / len(confidences)
    return avg_confidence, confidences
