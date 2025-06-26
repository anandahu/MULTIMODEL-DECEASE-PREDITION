# Multi-Disease Prediction System

This project implements a multi-disease prediction system using deep learning techniques. The system is designed to classify medical images for various diseases, leveraging advanced neural network architectures and data augmentation techniques. 

## Key Features

- **Multi-Disease Classification**: The system can predict multiple diseases from medical images, including pneumonia, skin cancer, brain tumors, and COVID-19.
- **Deep Learning Models**: Utilizes pre-trained models such as DenseNet121, ResNet50, and EfficientNetB0 for feature extraction and classification.
- **Data Management**: Automatically downloads and organizes datasets from Kaggle, ensuring a streamlined workflow.
- **Data Augmentation**: Implements data augmentation techniques to enhance model robustness and improve generalization.
- **Model Training and Evaluation**: Provides a comprehensive training pipeline with error handling, model evaluation, and performance metrics.
- **Visualization Tools**: Includes tools for visualizing training history, sample images, and Grad-CAM heatmaps for interpreting model predictions.

## Installation

To set up the project, ensure you have Python installed, then install the required libraries using the following commands:

```bash
pip install opendatasets tensorflow matplotlib scikit-learn seaborn plotly opencv-python pillow pandas numpy
```

## Usage

1. **Training Models**: 
   - You can train models for specific diseases by modifying the `diseases_to_train` list in the Jupyter Notebook located at `src/MULTIMODAL_DECEASE__PRED.ipynb`.
   - Run the notebook to execute the training pipeline.

2. **Making Predictions**:
   - After training, you can load the trained models and make predictions on new images.
   - Use the provided methods in the `MultiDiseasePredictor` class to predict single or multiple diseases from an image.

3. **Visualizing Results**:
   - The project includes visualization tools to plot training history and display Grad-CAM heatmaps for better understanding of model predictions.

## Example Usage

```python
# Load trained models
predictor = MultiDiseasePredictor()
predictor.load_models(['pneumonia', 'skin_cancer'])

# Predict single disease
result = predictor.predict_image('test_image.jpg', 'pneumonia')
print(f"Prediction: {result['predicted_class']} (Confidence: {result['confidence']:.2f})")

# Predict multiple diseases
results = predictor.predict_multiple_diseases('test_image.jpg')
for disease, result in results.items():
    print(f"{disease}: {result['predicted_class']} ({result['confidence']:.2f})")
```

## Conclusion

This multi-disease prediction system serves as a powerful tool for medical image classification, providing a robust framework for training and evaluating deep learning models. It is designed to be user-friendly, allowing users to easily train models and make predictions on new medical images.