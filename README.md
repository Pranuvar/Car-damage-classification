# 🚗 Car Damage Classification System

> Automated vehicle damage detection using transfer learning — VGG16, ResNet50, and EfficientNet benchmarked end-to-end.

---

## The Problem

Manual vehicle damage assessment is slow, inconsistent, and expensive. Insurance companies, fleet operators, and repair shops need a faster, more reliable way to identify and categorise damage from images. This project builds an automated classification pipeline that can process vehicle images and output damage assessments — removing the bottleneck of manual inspection.

---

## Key Results

| Metric | Value |
|---|---|
| **Architectures benchmarked** | VGG16 · ResNet50 · EfficientNet |
| **Training images** | ~970 (with augmentation) |
| **Approach** | Transfer learning on ImageNet weights |
| **Pipeline** | End-to-end: ingestion → augmentation → train → classify |

---

## Approach

Rather than training from scratch on a small dataset (~970 images), transfer learning was applied using models pre-trained on ImageNet. Fine-tuning on the domain-specific damage dataset allowed the models to leverage existing visual feature representations while adapting to the damage classification task.

Three architectures were benchmarked to identify the best fit for accuracy vs. inference speed:

| Architecture | Strengths |
|---|---|
| **VGG16** | Strong baseline; interpretable feature maps |
| **ResNet50** | Deeper with residual connections; handles vanishing gradient well |
| **EfficientNet** | Best accuracy-to-parameter ratio; fastest inference |

---

## Data Augmentation

With a limited ~970-image dataset, augmentation was critical to prevent overfitting:

- Horizontal and vertical flips
- Random rotation (±20°)
- Zoom and shear transformations
- Brightness adjustment
- Width/height shifts

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x · Keras |
| Architectures | VGG16 · ResNet50 · EfficientNet (ImageNet weights) |
| Image Processing | OpenCV · PIL |
| Data Augmentation | Keras ImageDataGenerator |
| Version Control | Git |

---

## How to Run

### Prerequisites
- Python 3.9+
- GPU recommended (CUDA-compatible), CPU workable for inference

```bash
git clone https://github.com/Pranuvar/car-damage-classification.git
cd car-damage-classification

pip install -r requirements.txt
```

### Training

```bash
# Organise your dataset:
# data/train/damaged/
# data/train/undamaged/
# data/val/damaged/
# data/val/undamaged/

python src/train.py --model vgg16       # or resnet50, efficientnet
```

### Inference on a new image

```bash
python src/predict.py --image path/to/image.jpg --model efficientnet
```

---

## Project Structure

```
car-damage-classification/
├── data/
│   ├── train/
│   └── val/
├── src/
│   ├── preprocess.py     # Data loading and augmentation
│   ├── model.py          # Architecture definitions and fine-tuning
│   ├── train.py          # Training loop and evaluation
│   └── predict.py        # Single-image inference
├── models/               # Saved weights per architecture
├── notebooks/            # Experimentation and visualisation
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why transfer learning?**
With ~970 images, training a deep CNN from random initialisation would almost certainly overfit. Transfer learning provides strong visual priors from ImageNet (14M images) that generalise well to damage detection.

**Why benchmark three architectures?**
Different architectures have different accuracy/speed trade-offs. EfficientNet consistently outperformed in accuracy-per-parameter, but VGG16 produced more interpretable intermediate activations — useful for understanding what the model is responding to.

---

## Author

**Praneeth Varma Danthuluri**
MSc Artificial Intelligence — National College of Ireland
[LinkedIn](https://linkedin.com/in/praneeth-varma-danthuluri) · [Portfolio](https://Pranuvar.github.io)
