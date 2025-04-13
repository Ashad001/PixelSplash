# PixelSplash: CNN-Driven Color Assault on Grayscale Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**PixelSplash** is a Computer Vision course assignment that takes a grayscale world and splashes it with AI-powered color. Using deep learning, this project explores how Convolutional Neural Networks (CNNs) can predict and reconstruct color in images—focusing on the 'Horse' class from the CIFAR-10 dataset.

I dived into two core paradigms in image colorization:
1. **Regression:** Predicting exact RGB values for every pixel.
2. **Classification:** Predicting from a set of quantized color classes.

Each paradigm is tested with two architectures:
- A **Standard Encoder-Decoder CNN**
- A **UNet** model with **skip connections** for spatial retention.

This assignment not only satisfies academic requirements but also pushes the creative and technical boundaries of colorization.

## 🎯 Fundamentals of Computer Vision – Assignment Project

This project was built as part of the **Fundamentals of Computer Vision** course. The objective was to practically implement, train, and evaluate image-to-image deep learning models, understanding how CNNs can reconstruct complex image features such as color.

Through this project, I applied foundational CV concepts like:
- Image preprocessing and augmentation
- Encoder-decoder architecture design
- Loss functions for pixel-wise learning
- Evaluation via perceptual and pixel-based metrics

---

## 🔥 Highlights

- Dual approaches: Regression vs. Classification-based color prediction.
- Comparative study: Standard CNN vs. UNet with skip connections.
- Visualizations of training, predictions, and performance metrics.
- Clean, modular implementation via Jupyter Notebooks.
- Built entirely using Python, Keras, NumPy, and Matplotlib.

---

## 📁 Project Structure

```
PixelSplash/
├── colorization_regression.ipynb      # Regression: CNN & UNet
├── colorization_classification.ipynb  # Classification: CNN & UNet
├── images/
│   ├── regression_generated_images.png
│   ├── regression_progress_report.png
│   ├── classification_generated_images.png
│   └── classification_progress_report.png
├── report/
│   ├── report.tex
│   ├── report.pdf
│   └── references.bib
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧠 Methodology

### 🔷 1. Regression

- **Goal:** Direct RGB prediction per pixel.
- **Models:** 
  - Basic CNN (Encoder-Decoder)
  - UNet CNN with skip connections
- **Loss:** MSE, MAE
- **Output:** Continuous 3-channel RGB image

### 🔶 2. Classification

- **Goal:** Classify each pixel into a color bin (cluster).
- **Preprocessing:** RGB → 24 color clusters via K-Means
- **Loss:** Categorical Cross-Entropy
- **Output:** Probability map → cluster center → RGB

---

## 🧪 Dataset

- **Source:** CIFAR-10
- **Focus:** 'Horse' class
- **Input:** Grayscale (simulated using Lab space L channel)
- **Target:** Original color (RGB or Lab ab channels)

---

## 📊 Results Summary

- **UNet > CNN:** UNet consistently outperformed the base CNN in both paradigms.
- **Skip Connections Matter:** Better detail reconstruction, smoother edges, and sharper textures.
- **Regression Wins Metrics:** Slightly better MSE, PSNR, and SSIM—but classification gave more vibrant results.
- **Visual Splash:** Regression was more accurate; classification more creative.

---

## 🚀 Getting Started

```bash
git clone 
cd PixelSplash

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
# or
jupyter lab
```

---

## 📦 Dependencies

- Python 3.7+
- TensorFlow / Keras
- NumPy
- Scikit-learn (for K-Means)
- Matplotlib
- OpenCV-Python (optional)
- Jupyter

---

## 🔮 Future Enhancements

- Use perceptual loss functions (VGG-based).
- Apply Lab color space for better human perception.
- Experiment with attention mechanisms or GANs.
- Extend beyond 'Horse' class or use larger datasets.
- Real-world colorization (e.g., historical photos).

---

## 📜 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

This project was completed as part of the **CS-AI Fundamentals of Computer Vision** course at FAST-NUCES, Karachi. Inspired by works in image-to-image translation and neural colorization literature.

---

> “Color is not just a visual property—it’s a neural prediction. PixelSplash brings that prediction to life.”