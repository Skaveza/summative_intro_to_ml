
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical

def validate_and_preprocess(data, prediction_mode=False):
    try:
        if isinstance(data, dict):
            df = pd.DataFrame(data.get('train', data))
        else:
            df = data.copy()

        # Required columns
        required_columns = ['services', 'operating_hours']
        if not prediction_mode:
            required_columns.append('rating')

        # Validation
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Handle missing ratings (only for training)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            rating_mean = df['rating'].mean()
            df['rating'] = df['rating'].fillna(rating_mean)
        else:
            rating_mean = None

        return df, rating_mean

    except Exception as e:
        raise Exception(f"Data validation error: {str(e)}")


def prepare_categorical_features(df, preprocessors):
    try:
        categorical_cols = ['care_system', 'mode of payment', 'Subcounty']

        if 'categorical_encoder' not in preprocessors or preprocessors['categorical_encoder'] is None:
            preprocessors['categorical_encoder'] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = preprocessors['categorical_encoder'].fit_transform(df[categorical_cols])
        else:
            encoded_data = preprocessors['categorical_encoder'].transform(df[categorical_cols])

        # Get feature names
        feature_names = []
        for i, col in enumerate(categorical_cols):
            categories = preprocessors['categorical_encoder'].categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])

        return pd.DataFrame(encoded_data, columns=feature_names)

    except Exception as e:
        raise Exception(f"Categorical feature error: {str(e)}")


def prepare_text_features(df, text_columns, text_processors=None, max_features=100):
    try:
        if text_processors is None:
            text_processors = {}

        features = []
        feature_names = []

        for col in text_columns:
            if col not in df.columns:
                continue

            if col not in text_processors:
                text_processors[col] = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                text_features = text_processors[col].fit_transform(df[col].fillna(''))
            else:
                text_features = text_processors[col].transform(df[col].fillna(''))

            features.append(text_features.toarray())
            feature_names.extend([f"{col}_tfidf_{i}" for i in range(text_features.shape[1])])

        combined_features = np.hstack(features) if features else np.array([])

        return {
            'features': combined_features,
            'processors': text_processors,
            'feature_names': feature_names
        }

    except Exception as e:
        raise Exception(f"Text processing error: {str(e)}")


def prepare_training_data(df, text_max_features=100):
    try:
        preprocessors = {
            'text_processors': {},
            'categorical_encoder': None,
            'feature_names': [],
            'scaler': None,
            'label_encoder': None,
            'rating_bins': None,
            'labels': None
        }

        # Process categorical and text features
        categorical_encoded = prepare_categorical_features(df, preprocessors)
        text_features = prepare_text_features(df, ['services', 'operating_hours'], preprocessors['text_processors'], text_max_features)

        # Process numerical features
        numerical_cols = ['latitude', 'longitude']
        num_data = df[numerical_cols].fillna(0).values

        # Combine all features
        X = np.hstack([num_data, categorical_encoded.values, text_features['features']])
        X = np.array(X)  # Ensure NumPy format

        # Apply scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Store feature names for consistency
        feature_names = numerical_cols + categorical_encoded.columns.tolist() + text_features['feature_names']
        preprocessors.update({'scaler': scaler, 'feature_names': feature_names})

        # Encode labels
        label_encoder = LabelEncoder()
        rating_bins = [-float('inf'), 2.5, 3.5, float('inf')]
        labels = ['Low', 'Medium', 'High']
        df['rating_category'] = pd.cut(df['rating'], bins=rating_bins, labels=labels)
        y_encoded = label_encoder.fit_transform(df['rating_category'])

        preprocessors.update({'label_encoder': label_encoder, 'rating_bins': rating_bins, 'labels': labels})

        # Handle class imbalance
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_scaled, y_encoded)

        # Save preprocessors
        preprocessor_save_path = "/content/drive/MyDrive/saved_preprocessors/preprocessors_latest.pkl"
        joblib.dump(preprocessors, preprocessor_save_path)
        print(f"Preprocessors saved successfully at: {preprocessor_save_path}")

        return {
            'X': X_scaled,
            'y': to_categorical(y_encoded),
            'X_resampled': X_resampled,
            'y_resampled': to_categorical(y_resampled),
            'preprocessors': preprocessors,
            'y_cat': y_encoded
        }

    except Exception as e:
        raise Exception(f"Training data preparation failed: {str(e)}")


def prepare_prediction_data(input_data, preprocessors):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
        if not preprocessors:
            raise Exception("Preprocessors dictionary is missing. Ensure training was completed successfully.")

        # 1. Process each feature type
        processed_features = []

        # Numerical features
        numerical_cols = ['latitude', 'longitude']
        num_data = pd.DataFrame()
        for col in numerical_cols:
            num_data[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        processed_features.append(num_data)

        # Categorical features
        if 'categorical_encoder' in preprocessors and preprocessors['categorical_encoder'] is not None:
            cat_data = preprocessors['categorical_encoder'].transform(
                df[preprocessors['categorical_encoder'].feature_names_in_]
            )
            processed_features.append(pd.DataFrame(cat_data))
        else:
            raise Exception("Categorical encoder is missing or was not fitted. Ensure training was completed successfully.")

        # Text features
        if 'text_processors' in preprocessors:
            for col, vectorizer in preprocessors['text_processors'].items():
                if col in df.columns:
                    tfidf_data = vectorizer.transform(df[col].fillna(''))
                    processed_features.append(pd.DataFrame(tfidf_data.toarray()))
                else:
                    raise Exception(f"Text processor for column '{col}' is missing. Ensure training was completed successfully.")

        # 2. Combine and align features
        X = pd.concat(processed_features, axis=1)

        # Remove duplicate columns from X
        X = X.loc[:, ~X.columns.duplicated()]

        # Ensure feature alignment
        if 'feature_names' not in preprocessors or not preprocessors['feature_names']:
            raise Exception("Feature names missing. Ensure training was completed successfully.")

        # Ensure preprocessors['feature_names'] is unique
        unique_feature_names = list(dict.fromkeys(preprocessors['feature_names']))  # Removes duplicates

        # Debugging: Print feature name stats
        print(f"Total feature names: {len(preprocessors['feature_names'])}")
        print(f"Unique feature names: {len(unique_feature_names)}")
        print(f"Columns in X before reindexing: {len(X.columns)}")

        # Ensure X does not have duplicate column names
        X = X.loc[:, ~X.columns.duplicated()]

        # Now safely reindex
        X = X.reindex(columns=unique_feature_names, fill_value=0)

        # 3. Apply scaling
        if 'scaler' not in preprocessors or preprocessors['scaler'] is None:
            raise Exception("Scaler is missing or was not fitted. Ensure training was completed successfully.")

        X_array = X.to_numpy()
        scaled_X = preprocessors['scaler'].transform(X_array)
        return scaled_X

    except Exception as e:
        raise Exception(f"Prediction data preparation failed: {str(e)}")

