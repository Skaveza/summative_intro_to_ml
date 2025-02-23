# Healthcare Facility Rating Prediction

## Problem Statement
Access to quality healthcare remains a significant challenge for displaced individuals. Many face barriers that leave them vulnerable and without adequate care, making it difficult to receive timely and appropriate medical support. Healthcare is a fundamental human right, yet disparities in accessibility, availability, and service quality persist. This project leverages machine learning to predict healthcare facility ratings, improving decision-making for displaced populations and policymakers.

## Dataset Overview
This project utilizes a dataset of over 6,000 healthcare facilities across Uganda. The dataset includes:
- **Facility Name**: Official name of the healthcare facility.
- **Services Offered**: Medical services provided (e.g., maternity, emergency, outpatient).
- **Geolocation**: Latitude & Longitude.
- **Rating**: Quality score based on user feedback and facility performance.
- **Operating Hours**: Facility’s working schedule and availability.
- **Website & Contact Information**: Online presence and phone numbers.
- **Care System**: Healthcare model (public, private, NGO-run).
- **Payment Methods**: Accepted payment modes.
- **Subcounty**: Administrative region.

## Discussion of Findings

| Training Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|-----------|-------------|--------|---------------|--------|--------------|----------|----------|--------|-----------|
| Instance 1 | None | Default | No | 4 | 0.001 | 0.71 | 0.7033 | 0.71 | 0.713 |
| Instance 2 | Adam | L1 | 40 | Yes | 5 | 0.01 | 0.58 | 0.5633 | 0.58 | 0.57 |
| Instance 3 | Adam | L1+L2 | 100 | Yes | 5 | 0.005 | 0.71 | 0.7033 | 0.71 | 0.71 |
| Instance 4 | Adam | L1+L2 | 100 | Yes | 5 | 0.001 | **0.87** | **0.8633** | **0.87** | **0.88** |

**Model Accuracy: 73.71%**

### Summary of Best Performing Combination
The best performing model was **Instance 4**, which used:
- **Adam Optimizer**
- **L1 + L2 Regularization**
- **100 Epochs with Early Stopping**
- **5 Layers**
- **Learning Rate: 0.001**
- **Accuracy: 87%**

## Machine Learning vs Neural Network Implementation
While traditional machine learning models were tested, the neural network model performed better. The best hyperparameter tuning included:
- **Adam optimizer** for adaptive learning.
- **L1 + L2 regularization** to prevent overfitting.
- **Early stopping** to avoid unnecessary epochs.
- **Lower learning rates (0.001)** that improved stability.

The neural network outperformed standard machine learning algorithms due to its ability to capture complex patterns in the data, making it more suitable for healthcare facility rating predictions.

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/yourrepository.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model:
   ```sh
   python train_model.py
   ```
4. Make predictions:
   ```sh
   python predict.py
   ```

## Contributors
- Sifa Mwachoni

## License
This project is licensed under the MIT License.

