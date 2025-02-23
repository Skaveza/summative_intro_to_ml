# README: Healthcare Facility Rating Prediction

## Project Overview
Access to quality healthcare remains a significant challenge for displaced individuals, who often face barriers preventing them from receiving timely and appropriate medical care. This project leverages machine learning to predict healthcare facility ratings, assisting displaced individuals and policymakers in making informed healthcare decisions. By utilizing data-driven insights, we aim to bridge healthcare gaps and enhance service delivery for vulnerable populations.

## Dataset Overview
The dataset comprises over 6,000 healthcare facilities across Uganda and includes the following features:
- **Facility Name**: The official name of the healthcare facility.
- **Services Offered**: Types of medical services provided (e.g., maternity, emergency, outpatient).
- **Geolocation**: Latitude and longitude coordinates of the facility.
- **Rating**: Quality score based on user feedback and facility performance.
- **Operating Hours**: Facility working schedule and availability.
- **Website**: Official online presence (if available).
- **Contact Information**: Phone numbers for inquiries and emergencies.
- **Care System**: Type of healthcare model (public, private, NGO-run, etc.).
- **Payment Methods**: Accepted payment modes for medical services.
- **Subcounty**: The administrative region where the facility is located.

## Model Performance Comparison

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|----------|------------|-------------|--------|----------------|--------|---------------|----------|----------|--------|-----------|
| Instance 1 | None | None | 10(default)| No | 4 | 0.001 | 0.71 | 0.7033 | 0.71 | 0.713 |
| Instance 2 | Adam | L1 | 40 | Yes | 5 | 0.01 | 0.58 | 0.5633 | 0.58 | 0.57 |
| Instance 3 | Adam | L1+L2 | 100 | Yes | 5 | 0.005 | 0.71 | 0.7033 | 0.71 | 0.71 |
| Instance 4 | Adam | L1+L2 | 100 | Yes | 5 | 0.001 | 0.87 | 0.8633 | 0.87 | 0.88 |

### Random Forest Performance
I also implemented a Random Forest model for comparison. The hyperparameters used were:
- **Number of Estimators**: 200
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Random State**: 42

#### Random Forest Results:
- **Accuracy**: 78.42%
- **Precision**: 80.12%
- **Recall**: 77.85%
- **F1 Score**: 78.95%

## Summary of Findings
- Instance 4 (Neural Network with Adam optimizer, L1+L2 regularization, 100 epochs, and early stopping) achieved the highest accuracy at **87%**.
- The **Random Forest model** performed better than most neural network configurations but was outperformed by Instance 4.
- **Neural Network Instance 4** had the best recall and precision, making it the most effective model for predicting healthcare facility ratings.
- Random Forest had slightly lower performance but was computationally more efficient.

## Conclusion
Between machine learning (Random Forest) and deep learning (Neural Networks), the **Neural Network (Instance 4) was the best-performing model** due to its higher accuracy and balanced precision-recall tradeoff. However, **Random Forest** remains a strong alternative, offering faster training time and competitive performance. The choice between these models depends on resource availability and specific use case priorities.

## Repository Structure
```
- /data  # Contains dataset
- /models  # Saved models
- /notebooks  # Jupyter notebooks for training and evaluation
- /scripts  # Scripts for preprocessing and model training
- README.md  # Project documentation
```

## How to Run the Project
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:  
   ```bash
   python scripts/train_model.py
   ```
4. Make predictions:  
   ```bash
   python scripts/predict.py --input sample_input.json
   ```

## Contributors
- Your Name (your.email@example.com)

## License
This project is licensed under the MIT License.

