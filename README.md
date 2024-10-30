# Multi-Label Classification with Single Positive Labels and C-Tran

This project focuses on improving dataset quality in multi-label classification tasks in computer vision. We address challenges such as low resolution, noise, label imbalance, and synonymous or mutually inclusive labels, aiming to create robust and efficient models with enhanced accuracy and fewer parameters.

## Project Overview

This research explores the following key components:
- [**Single Positive Labels**](https://arxiv.org/pdf/2106.09708): Handling multi-label classification with only single positive labels, utilizing advanced architectures and training techniques.
- [**C-Tran Model Enhancements**](https://arxiv.org/pdf/2011.14027): Experimenting with various model configurations and training strategies, including modifying loss functions, activation functions, and model architectures.
- **Combined Model Approach**: Integrating features from multiple models to improve test accuracy and robustness.
- **Image Processing Techniques**: Cleaning and improving dataset quality by removing unnecessary backgrounds and optimizing input images for better feature extraction.

## Methodology

### Single Positive Labels Approach
In this approach, we experimented with different loss functions, training modes, and activation functions to optimize performance. Models used include ResNet50, EfficientNetB7, and EfficientNetB0. The study achieved more high validation and test accuracy than the origin model, with key results as follows:

| Model           | Loss Function | Training Mode | Activation Function | mAP (Validation) | mAP (Test) |
|-----------------|---------------|---------------|---------------------|------------------|------------|
| ResNet50        | AN-LS         | End-to-End    | Sigmoid            | 24.2951         | 34.6173    |
| EfficientNetB7  | AN-LS         | Transfer      | Softmax            | 0.4960          | 1.6186     |
| EfficientNetB0  | AN-LS         | Transfer      | Softmax            | 0.4709          | 1.0321     |

### C-Tran Model Performance
The C-Tran model employs binary cross-entropy loss to enhance multi-label classification. Results are obtained by varying the feature extraction network, encoder layers, label masking, and activation functions, achieving promising results across configurations.

| Feature Extractor | Activation Function | Encoder Layers | Label Masking | Known Labels | Label State | Test Accuracy |
|-------------------|---------------------|----------------|---------------|--------------|-------------|---------------|
| ResNet101         | Sigmoid            | 3              | Yes           | 0            | Total       | 91.3          |
| MobileNetV2       | Softmax            | 3              | Yes           | 0            | Total       | 89.8          |
| MobileNetV2       | Sigmoid            | 2              | No            | 0            | Total       | 91.3          |

### Combined Model Results
The combined model integrates features from EfficientNet B0 and MobileNet V2 using configurations specified in Table 4.12. This ensemble approach boosts the model's robustness and accuracy.

| Feature Extractor | Loss Function | Test Accuracy |
|-------------------|---------------|---------------|
| EfficientNet B0   | ROLE          | 54.9          |
| MobileNet V2      | AN-LS         | 51            |
| MobileNet V2      | HU            | 47            |

The incoming debugged combined approach will give a better result, we hope!!!.

## Key Results

The research achieved impressive results, especially for configurations using the C-Tran model with ResNet101 and MobileNetV2, achieving a **91.3%** test accuracy in multiple settings.

- For the overview, please see [Slide](./DOC/Slide.pdf)
- For the details, please see [Report](./DOC/Report.pdf)

## Installation

To run this project, clone the repository:

```bash
git clone https://github.com/HoDangCao/multi-object-classification-based-on-deep-learning-model.git
cd multi-object-classification-based-on-deep-learning-model
```

## Usage

Please read files for
- [Only-Positive-Label](./SOURCE/Only_Positive_Label/Guide_to_use_and_run_files.txt) setup.
- [C-Tran](./SOURCE/C-Tranl/Guide_to_implement.txt) setup.

## Contact

For questions or collaboration, please contact [me](mailto:dangcaoho151202@gmail.com).

---
