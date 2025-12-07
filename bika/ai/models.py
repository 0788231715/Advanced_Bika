import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class FruitQualityPredictor:
    """AI model for predicting fruit quality based on environmental conditions"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.fruit_encoder = LabelEncoder()
        
        # Features based on your dataset columns
        self.numerical_features = ['Temp', 'Humid (%)', 'Light (Fux)', 'CO2 (pmm)']
        self.categorical_features = ['Fruit']
        self.target_column = 'Class'
        
        self.class_names = ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
    
    def load_fruit_dataset(self, csv_path):
        """Load and prepare fruit quality dataset"""
        try:
            df = pd.read_csv(csv_path)
            
            # Rename columns to match your dataset
            column_mapping = {
                'Fruit': 'Fruit',
                'Temp': 'temperature',
                'Humid (%)': 'humidity',
                'Light (Fux)': 'light_intensity',
                'CO2 (pmm)': 'co2_level',
                'Class': 'quality_class'
            }
            
            # Rename columns for consistency
            df = df.rename(columns=column_mapping)
            
            # Clean data
            df = df.dropna()
            
            # Validate quality classes
            valid_classes = ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
            invalid_rows = df[~df['quality_class'].isin(valid_classes)]
            
            if len(invalid_rows) > 0:
                print(f"Warning: Found {len(invalid_rows)} rows with invalid quality classes")
                df = df[df['quality_class'].isin(valid_classes)]
            
            # Encode target variable
            df['quality_class_encoded'] = self.label_encoder.fit_transform(df['quality_class'])
            
            # Prepare features
            X = df[['Fruit', 'temperature', 'humidity', 'light_intensity', 'co2_level']].copy()
            y = df['quality_class_encoded'].values
            
            # Create preprocessing pipeline
            self._create_preprocessor(X)
            
            return X, y, df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None
    
    def _create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        from sklearn.impute import SimpleImputer
        
        # Numerical features preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, ['temperature', 'humidity', 'light_intensity', 'co2_level']),
                ('cat', categorical_transformer, ['Fruit'])
            ]
        )
        
        # Fit preprocessor
        self.preprocessor.fit(X)
    
    def train_model(self, X, y, test_size=0.2, cv_folds=5):
        """Train the selected model with cross-validation"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Preprocess features
        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
            
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                subsample=0.8
            )
            
        elif self.model_type == 'neural_network':
            self.model = self._create_neural_network(X_train_processed.shape[1])
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Cross-validation
        if self.model_type != 'neural_network':
            cv_scores = cross_val_score(
                self.model, X_train_processed, y_train,
                cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train model
        if self.model_type == 'neural_network':
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.model.fit(
                X_train_processed, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            self.model.fit(X_train_processed, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate detailed classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': list(self.label_encoder.classes_)
        }
    
    def _create_neural_network(self, input_dim):
        """Create neural network for fruit quality prediction"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_quality(self, fruit_type, temperature, humidity, light_intensity, co2_level):
        """Predict fruit quality for given conditions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'Fruit': fruit_type,
            'temperature': float(temperature),
            'humidity': float(humidity),
            'light_intensity': float(light_intensity),
            'co2_level': float(co2_level)
        }])
        
        # Preprocess
        processed_data = self.preprocessor.transform(input_data)
        
        # Make prediction
        if self.model_type == 'neural_network':
            predictions = self.model.predict(processed_data, verbose=0)
            predicted_class_idx = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
        else:
            predicted_class_idx = self.model.predict(processed_data)
            
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(processed_data), axis=1)
            else:
                confidence = np.ones(len(predicted_class_idx))
        
        # Decode predictions
        predicted_class = self.label_encoder.inverse_transform(predicted_class_idx)
        
        # Get probabilities for all classes
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_data)[0]
            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        else:
            class_probabilities = {}
        
        return {
            'predicted_class': predicted_class[0],
            'confidence': float(confidence[0]),
            'class_probabilities': class_probabilities,
            'input_conditions': {
                'fruit': fruit_type,
                'temperature': temperature,
                'humidity': humidity,
                'light_intensity': light_intensity,
                'co2_level': co2_level
            }
        }
    
    def predict_batch_quality(self, batch_data):
        """Predict quality for multiple readings"""
        predictions = []
        
        for data in batch_data:
            pred = self.predict_quality(
                data['fruit'],
                data['temperature'],
                data['humidity'],
                data['light_intensity'],
                data['co2_level']
            )
            predictions.append(pred)
        
        return predictions
    
    def save_model(self, model_path):
        """Save trained model and preprocessor"""
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'class_names': self.class_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.class_names = model_data['class_names']
        print(f"Model loaded from {model_path}")
        return True

class FruitRipenessPredictor:
    """Predict fruit ripeness based on ethylene and temperature"""
    
    def __init__(self):
        self.model = None
    
    def predict_ripeness(self, fruit_type, temperature, ethylene_level, days_since_harvest):
        """Predict ripeness level"""
        # Simple rule-based system - can be enhanced with ML
        ripeness_factors = {
            'temperature': temperature,
            'ethylene': ethylene_level,
            'days': days_since_harvest
        }
        
        # Different fruits have different ripening patterns
        ripening_rates = {
            'Banana': 0.8,
            'Apple': 0.3,
            'Orange': 0.4,
            'Mango': 1.0,
            'Tomato': 0.9,
            'Avocado': 0.7
        }
        
        base_rate = ripening_rates.get(fruit_type, 0.5)
        
        # Calculate ripeness score
        ripeness_score = min(1.0, base_rate * (1 + ethylene_level/100) * (1 + days_since_harvest/10))
        
        if ripeness_score < 0.3:
            return 'unripe'
        elif ripeness_score < 0.7:
            return 'ripe'
        else:
            return 'overripe'
    
    def estimate_shelf_life(self, fruit_type, current_quality, temperature, humidity):
        """Estimate remaining shelf life"""
        # Base shelf life for different fruits at optimal conditions
        base_shelf_life = {
            'Banana': 7,
            'Apple': 30,
            'Orange': 21,
            'Mango': 10,
            'Tomato': 14,
            'Avocado': 7,
            'Strawberry': 5,
            'Grapes': 14,
            'Watermelon': 21
        }
        
        base_days = base_shelf_life.get(fruit_type, 10)
        
        # Adjust based on conditions
        temp_factor = 1.0
        if temperature > 10:
            temp_factor = max(0.3, 1.0 - (temperature - 10) * 0.05)
        
        humidity_factor = 1.0
        if humidity < 85:
            humidity_factor = max(0.5, humidity / 85)
        
        # Adjust based on current quality
        quality_factor = {
            'Fresh': 1.0,
            'Good': 0.8,
            'Fair': 0.6,
            'Poor': 0.3,
            'Rotten': 0.0
        }.get(current_quality, 0.5)
        
        estimated_days = base_days * temp_factor * humidity_factor * quality_factor
        
        return max(1, int(estimated_days))

class EthyleneMonitor:
    """Monitor ethylene production and effects"""
    
    def __init__(self):
        self.ethylene_producers = ['Apple', 'Banana', 'Tomato', 'Avocado', 'Pear']
        self.ethylene_sensitive = ['Lettuce', 'Carrot', 'Broccoli', 'Cucumber', 'Watermelon']
    
    def check_compatibility(self, fruit1, fruit2):
        """Check if two fruits can be stored together"""
        if fruit1 in self.ethylene_producers and fruit2 in self.ethylene_sensitive:
            return False
        if fruit2 in self.ethylene_producers and fruit1 in self.ethylene_sensitive:
            return False
        return True
    
    def get_ethylene_management_tips(self, fruit_type):
        """Get tips for ethylene management"""
        tips = []
        
        if fruit_type in self.ethylene_producers:
            tips.append(f"{fruit_type} produces ethylene - store separately from ethylene-sensitive produce")
            tips.append("Use ethylene absorbers in storage")
            tips.append("Maintain good ventilation")
        
        if fruit_type in self.ethylene_sensitive:
            tips.append(f"{fruit_type} is ethylene-sensitive - keep away from ethylene producers")
            tips.append("Store in separate compartments")
            tips.append("Monitor for premature ripening")
        
        return tips