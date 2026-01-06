import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pretained_model import PretrainedModel
from data.image_loader import load_images_from_folder
from evaluation.baseline_eval import evaluate_baseline
from evaluation.failure_eval import evaluate_failure
from failure_generators.blur import apply_gaussian_blur

def main():
    print("Loading pretrained model...")
    model = PretrainedModel()

    print("Loading clean images...")
    clean_images, names = load_images_from_folder("data/original")

    if len(clean_images) == 0:
        print("No images found in data/original")
        return

    print(f"Loaded {len(clean_images)} images")

    # BASELINE
    print("\nRunning baseline evaluation...")
    baseline_avg, _ = evaluate_baseline(model, clean_images)
    print(f"Baseline avg confidence: {baseline_avg:.4f}")

    # APPLY BLUR
    print("\nApplying Gaussian blur...")
    blurred_images = [apply_gaussian_blur(img, kernel_size=15) for img in clean_images]

    # FAILURE EVALUATION
    print("\nEvaluating failure impact...")
    avg_drop, drops = evaluate_failure(model, clean_images, blurred_images)

    print(f"Average confidence drop due to blur: {avg_drop:.4f}")

if __name__ == "__main__":
    main()
