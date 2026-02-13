# bika/ai_models.py - COMPREHENSIVE AI MODELS FOR FRUIT QUALITY MONITORING
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

warnings.filterwarnings('ignore')

# ==================== DEPENDENCIES WITH GRACEFUL FALLBACKS ====================

# Try to import scikit-learn
try:
    from sklearn.ensemble import (
        IsolationForest, RandomForestRegressor, RandomForestClassifier, 
        GradientBoostingClassifier, AdaBoostClassifier
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        mean_squared_error, mean_absolute_error, r2_score,
        precision_score, recall_score, f1_score
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some AI features will be disabled.")

# Try to import joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Model saving/loading will not work.")

# Try to import xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available.")

# Try to import tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input,
        LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural networks disabled.")

# Try to import statsmodels for time series
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Time series forecasting disabled.")

# ==================== CORE AI MODELS ====================

class FruitQualityPredictor:
    """AI model for predicting fruit quality based on environmental conditions"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.class_names = ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
        self.feature_columns = ['temperature', 'humidity', 'light_intensity', 'co2_level', 'fruit_type']
        self.model_metrics = {}
        
    def load_fruit_dataset(self, csv_path, target_column='quality_class'):
        """Load and prepare fruit quality dataset with validation"""
        if not SKLEARN_AVAILABLE:
            return None, None, None
            
        try:
            df = pd.read_csv(csv_path)
            
            # Dataset validation and cleaning
            required_columns = ['temperature', 'humidity', 'light_intensity', 'co2_level', 'fruit_type', target_column]
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns in dataset: {missing_cols}")
                return None, None, None
            
            # Clean data
            df = df.dropna()
            
            # Remove outliers using IQR method
            numeric_cols = ['temperature', 'humidity', 'light_intensity', 'co2_level']
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Validate quality classes
            valid_classes = self.class_names
            invalid_rows = df[~df[target_column].isin(valid_classes)]
            
            if len(invalid_rows) > 0:
                print(f"Warning: Found {len(invalid_rows)} rows with invalid quality classes")
                df = df[df[target_column].isin(valid_classes)]
            
            if len(df) < 10:
                print(f"Warning: Insufficient data after cleaning. Only {len(df)} samples remaining.")
                return None, None, None
            
            # Encode target variable
            df['quality_class_encoded'] = self.label_encoder.fit_transform(df[target_column])
            
            # Prepare features
            X = df[self.feature_columns].copy()
            y = df['quality_class_encoded'].values
            
            # Create preprocessing pipeline
            self._create_preprocessor(X)
            
            return X, y, df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None
    
    def _create_preprocessor(self, X):
        """Create preprocessing pipeline for features"""
        if not SKLEARN_AVAILABLE:
            return
            
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
                ('cat', categorical_transformer, ['fruit_type'])
            ]
        )
        
        # Fit preprocessor
        self.preprocessor.fit(X)
    
    def select_best_model(self, X_train, y_train):
        """Select the best model using grid search"""
        if not SKLEARN_AVAILABLE:
            return None
        
        # Define model candidates
        model_candidates = [
            {
                'name': 'random_forest',
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            {
                'name': 'gradient_boosting',
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            {
                'name': 'svm',
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            {
                'name': 'knn',
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            }
        ]
        
        best_score = -1
        best_model = None
        best_name = 'random_forest'
        
        for candidate in model_candidates:
            try:
                print(f"\nTraining {candidate['name']}...")
                grid_search = GridSearchCV(
                    candidate['model'],
                    candidate['params'],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_name = candidate['name']
                    
                print(f"Best {candidate['name']} accuracy: {grid_search.best_score_:.4f}")
                print(f"Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"Error training {candidate['name']}: {e}")
                continue
        
        print(f"\nSelected best model: {best_name} with accuracy: {best_score:.4f}")
        return best_model, best_name
    
    def train_model(self, X, y, test_size=0.2, cv_folds=5, use_grid_search=True):
        """Train the model with comprehensive evaluation"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
            
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Preprocess features
            X_train_processed = self.preprocessor.transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Model selection
            if use_grid_search:
                self.model, self.model_type = self.select_best_model(X_train_processed, y_train)
                if self.model is None:
                    # Fallback to default
                    self.model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'
                    )
                    self.model_type = 'random_forest'
            else:
                # Use specified model type
                if self.model_type == 'random_forest':
                    self.model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'
                    )
                elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
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
                elif self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
                    self.model = self._create_neural_network(X_train_processed.shape[1])
                else:
                    self.model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'
                    )
                    self.model_type = 'random_forest'
            
            # Train model
            if self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
                
                self.model.fit(
                    X_train_processed, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
            else:
                self.model.fit(X_train_processed, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_processed)
            y_pred_proba = self.model.predict_proba(X_test_processed) if hasattr(self.model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_processed, y_train,
                cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            
            # Detailed classification report
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_.tolist()
            
            self.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': self.model_type,
                'class_names': list(self.label_encoder.classes_)
            }
            
            return self.model_metrics
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {'error': str(e)}
    
    def _create_neural_network(self, input_dim):
        """Create neural network for fruit quality prediction"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
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
        """Predict fruit quality with confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        try:
            # Prepare input data
            input_data = pd.DataFrame([{
                'fruit_type': fruit_type,
                'temperature': float(temperature),
                'humidity': float(humidity),
                'light_intensity': float(light_intensity),
                'co2_level': float(co2_level)
            }])
            
            # Preprocess
            processed_data = self.preprocessor.transform(input_data)
            
            # Make prediction
            if self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
                predictions = self.model.predict(processed_data, verbose=0)
                predicted_class_idx = np.argmax(predictions, axis=1)
                confidence = np.max(predictions, axis=1)
                probabilities = predictions[0]
            else:
                predicted_class_idx = self.model.predict(processed_data)
                
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(processed_data)[0]
                    confidence = np.max(probabilities)
                else:
                    confidence = 1.0
                    probabilities = np.ones(len(self.label_encoder.classes_)) / len(self.label_encoder.classes_)
            
            # Decode predictions
            predicted_class = self.label_encoder.inverse_transform(predicted_class_idx)
            
            # Get class probabilities
            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            # Calculate quality score (0-100)
            quality_scores = {'Fresh': 100, 'Good': 80, 'Fair': 60, 'Poor': 30, 'Rotten': 0}
            quality_score = quality_scores.get(predicted_class[0], 50)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                predicted_class[0], temperature, humidity, light_intensity, co2_level
            )
            
            return {
                'predicted_class': predicted_class[0],
                'confidence': float(confidence),
                'quality_score': quality_score,
                'class_probabilities': class_probabilities,
                'recommendations': recommendations,
                'input_conditions': {
                    'fruit': fruit_type,
                    'temperature': temperature,
                    'humidity': humidity,
                    'light_intensity': light_intensity,
                    'co2_level': co2_level
                }
            }
            
        except Exception as e:
            print(f"Error predicting quality: {e}")
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'quality_score': 0,
                'class_probabilities': {},
                'recommendations': ['Error in prediction'],
                'input_conditions': {}
            }
    
    def _generate_recommendations(self, quality_class, temp, humidity, light, co2):
        """Generate recommendations based on predicted quality"""
        recommendations = []
        
        # Temperature recommendations
        if temp < 2:
            recommendations.append("Increase temperature to 2-8°C range")
        elif temp > 12:
            recommendations.append("Decrease temperature to 2-8°C range")
        
        # Humidity recommendations
        if humidity < 85:
            recommendations.append("Increase humidity to 85-95% range")
        elif humidity > 95:
            recommendations.append("Decrease humidity to 85-95% range")
        
        # Light recommendations
        if light > 100:
            recommendations.append("Reduce light exposure below 100 lux")
        
        # CO₂ recommendations
        if co2 > 1000:
            recommendations.append("Improve ventilation to reduce CO₂ levels")
        
        # Quality-based recommendations
        if quality_class in ['Poor', 'Rotten']:
            recommendations.append("Consider immediate processing or discount sale")
            recommendations.append("Check for ethylene exposure from other fruits")
        
        return recommendations
    
    def save_model(self, model_path):
        """Save trained model and preprocessor"""
        if not JOBLIB_AVAILABLE:
            print(f"Warning: joblib not available. Model not saved to {model_path}")
            return
        
        try:
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'class_names': self.class_names,
                'feature_columns': self.feature_columns,
                'model_metrics': self.model_metrics,
                'model_info': {
                    'created_at': datetime.now().isoformat(),
                    'trained_samples': self.model_metrics.get('training_samples', 0),
                    'accuracy': self.model_metrics.get('accuracy', 0)
                }
            }
            
            joblib.dump(model_data, model_path)
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, model_path):
        """Load trained model"""
        if not JOBLIB_AVAILABLE:
            print(f"Error: joblib not available. Cannot load model from {model_path}")
            return False
        
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.class_names = model_data['class_names']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data.get('model_metrics', {})
            
            print(f"Model loaded from {model_path}")
            print(f"Model info: {model_data.get('model_info', {})}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def explain_prediction(self, fruit_type, temperature, humidity, light_intensity, co2_level):
        """Explain the prediction using feature importance"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        try:
            # Get feature names
            feature_names = []
            
            # Numerical features
            feature_names.extend(['temperature', 'humidity', 'light_intensity', 'co2_level'])
            
            # Categorical features (fruit types)
            if hasattr(self.preprocessor, 'named_transformers_'):
                fruit_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                fruit_categories = fruit_encoder.categories_[0]
                feature_names.extend([f'fruit_{cat}' for cat in fruit_categories])
            
            # Get feature importances
            if len(feature_names) == len(self.model.feature_importances_):
                importances = list(zip(feature_names, self.model.feature_importances_))
                importances.sort(key=lambda x: x[1], reverse=True)
                
                # Filter to show only relevant features for this prediction
                explanation = {
                    'top_features': importances[:5],
                    'fruit_type_importance': next(
                        (imp for name, imp in importances if f'fruit_{fruit_type}' in name),
                        0.0
                    ),
                    'temperature_importance': next(
                        (imp for name, imp in importances if 'temperature' in name),
                        0.0
                    )
                }
                
                return explanation
            
        except Exception as e:
            print(f"Error explaining prediction: {e}")
        
        return None


class FruitRipenessPredictor:
    """Predict fruit ripeness based on multiple factors"""
    
    def __init__(self):
        self.ripening_models = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default ripening models for common fruits"""
        # Ripening rates (higher = faster ripening)
        self.ripening_rates = {
            'Banana': {'base_rate': 0.8, 'ethylene_factor': 1.5, 'temp_factor': 0.1},
            'Apple': {'base_rate': 0.3, 'ethylene_factor': 1.2, 'temp_factor': 0.05},
            'Orange': {'base_rate': 0.4, 'ethylene_factor': 1.1, 'temp_factor': 0.08},
            'Mango': {'base_rate': 1.0, 'ethylene_factor': 2.0, 'temp_factor': 0.15},
            'Tomato': {'base_rate': 0.9, 'ethylene_factor': 1.8, 'temp_factor': 0.12},
            'Avocado': {'base_rate': 0.7, 'ethylene_factor': 1.6, 'temp_factor': 0.1},
            'Strawberry': {'base_rate': 0.5, 'ethylene_factor': 1.3, 'temp_factor': 0.07},
            'Grapes': {'base_rate': 0.4, 'ethylene_factor': 1.1, 'temp_factor': 0.06},
            'Watermelon': {'base_rate': 0.3, 'ethylene_factor': 1.0, 'temp_factor': 0.04},
            'Pineapple': {'base_rate': 0.2, 'ethylene_factor': 1.0, 'temp_factor': 0.03},
        }
        
        # Shelf life at optimal conditions (days)
        self.base_shelf_life = {
            'Banana': 7,
            'Apple': 30,
            'Orange': 21,
            'Mango': 10,
            'Tomato': 14,
            'Avocado': 7,
            'Strawberry': 5,
            'Grapes': 14,
            'Watermelon': 21,
            'Pineapple': 10,
        }
    
    def predict_ripeness(self, fruit_type, temperature, ethylene_level, days_since_harvest, 
                         humidity=None, light_exposure=None):
        """Predict ripeness level with multiple factors"""
        fruit_type = fruit_type.capitalize()
        
        # Get ripening parameters
        params = self.ripening_rates.get(fruit_type, {'base_rate': 0.5, 'ethylene_factor': 1.2, 'temp_factor': 0.08})
        
        # Calculate ripening score (0-1)
        base_rate = params['base_rate']
        ethylene_factor = 1 + (params['ethylene_factor'] * ethylene_level / 100)
        
        # Temperature effect (optimal around 20°C)
        temp_effect = 1 + (params['temp_factor'] * abs(temperature - 20) / 10)
        
        # Days effect
        days_effect = 1 + (days_since_harvest / 10)
        
        # Humidity effect (if provided)
        humidity_effect = 1.0
        if humidity:
            if humidity < 80:
                humidity_effect = 1.1  # Faster ripening in dry conditions
            elif humidity > 95:
                humidity_effect = 0.9  # Slower in very humid
        
        # Light effect (if provided)
        light_effect = 1.0
        if light_exposure:
            if light_exposure > 100:
                light_effect = 1.2  # Faster with light
        
        ripeness_score = min(1.0, base_rate * ethylene_factor * temp_effect * days_effect * humidity_effect * light_effect)
        
        # Determine ripeness stage
        if ripeness_score < 0.3:
            stage = 'unripe'
            color = 'green'
        elif ripeness_score < 0.6:
            stage = 'ripe'
            color = 'yellow'
        elif ripeness_score < 0.8:
            stage = 'fully_ripe'
            color = 'orange'
        else:
            stage = 'overripe'
            color = 'brown'
        
        return {
            'ripeness_stage': stage,
            'ripeness_score': ripeness_score,
            'color_indicator': color,
            'estimated_days_to_overripe': max(0, int((1.0 - ripeness_score) * 3)),
            'factors': {
                'temperature_effect': temp_effect,
                'ethylene_effect': ethylene_factor,
                'days_effect': days_effect,
                'humidity_effect': humidity_effect,
                'light_effect': light_effect
            }
        }
    
    def estimate_shelf_life(self, fruit_type, current_quality, temperature, humidity, 
                           ethylene_present=False, storage_conditions='optimal'):
        """Estimate remaining shelf life in days"""
        fruit_type = fruit_type.capitalize()
        
        # Base shelf life
        base_days = self.base_shelf_life.get(fruit_type, 10)
        
        # Quality factor
        quality_factor = {
            'Fresh': 1.0,
            'Good': 0.8,
            'Fair': 0.6,
            'Poor': 0.3,
            'Rotten': 0.0
        }.get(current_quality, 0.5)
        
        # Temperature adjustment (optimal around 4°C for most fruits)
        if temperature <= 4:
            temp_factor = 1.0  # Optimal refrigeration
        elif temperature <= 10:
            temp_factor = 0.8
        elif temperature <= 20:
            temp_factor = 0.5
        else:
            temp_factor = 0.2  # Very short shelf life at room temperature
        
        # Humidity adjustment
        if humidity is not None:
            if 85 <= humidity <= 95:
                humidity_factor = 1.0  # Optimal
            elif 70 <= humidity < 85:
                humidity_factor = 0.7
            elif humidity < 70:
                humidity_factor = 0.5  # Drying out
            else:
                humidity_factor = 0.8  # Too humid
        else:
            humidity_factor = 1.0
        
        # Ethylene presence
        ethylene_factor = 0.7 if ethylene_present else 1.0
        
        # Storage conditions
        storage_factor = {
            'optimal': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.3
        }.get(storage_conditions, 0.5)
        
        # Calculate estimated days
        estimated_days = base_days * quality_factor * temp_factor * humidity_factor * ethylene_factor * storage_factor
        
        # Round to nearest half day
        estimated_days = max(0.5, round(estimated_days * 2) / 2)
        
        return {
            'estimated_days': estimated_days,
            'base_days': base_days,
            'factors': {
                'quality': quality_factor,
                'temperature': temp_factor,
                'humidity': humidity_factor,
                'ethylene': ethylene_factor,
                'storage': storage_factor
            },
            'recommendations': self._get_shelf_life_recommendations(
                fruit_type, temperature, humidity, ethylene_present
            )
        }
    
    def _get_shelf_life_recommendations(self, fruit_type, temperature, humidity, ethylene_present):
        """Get recommendations to extend shelf life"""
        recommendations = []
        
        # Temperature recommendations
        if temperature > 10:
            recommendations.append(f"Store {fruit_type} at lower temperature (4-8°C recommended)")
        
        # Humidity recommendations
        if humidity and humidity < 85:
            recommendations.append(f"Increase humidity to 85-95% for {fruit_type}")
        elif humidity and humidity > 95:
            recommendations.append(f"Reduce humidity to prevent mold growth")
        
        # Ethylene recommendations
        if ethylene_present:
            recommendations.append(f"Store {fruit_type} away from ethylene-producing fruits")
        
        # Fruit-specific recommendations
        fruit_specific = {
            'Banana': ['Store at room temperature until ripe, then refrigerate'],
            'Tomato': ['Never refrigerate - store at room temperature'],
            'Avocado': ['Store at room temperature until ripe, then refrigerate'],
            'Apple': ['Can be stored in refrigerator for months'],
            'Orange': ['Store in cool, well-ventilated area'],
        }
        
        recommendations.extend(fruit_specific.get(fruit_type, []))
        
        return recommendations


class EthyleneMonitor:
    """Monitor ethylene production and effects on fruits, acting as a predictor."""
    
    def __init__(self):
        self.ethylene_producers = {
            'Apple': {'level': 'high', 'production_rate': 100}, # µL/kg/hour
            'Banana': {'level': 'high', 'production_rate': 100},
            'Tomato': {'level': 'high', 'production_rate': 100},
            'Avocado': {'level': 'high', 'production_rate': 100},
            'Pear': {'level': 'medium', 'production_rate': 50},
            'Peach': {'level': 'medium', 'production_rate': 50},
            'Plum': {'level': 'medium', 'production_rate': 50},
            'Papaya': {'level': 'medium', 'production_rate': 50},
            'Mango': {'level': 'medium', 'production_rate': 50},
            'Kiwi': {'level': 'low', 'production_rate': 20},
        }
        
        self.ethylene_sensitive = {
            'Lettuce': 'very_high',
            'Broccoli': 'very_high',
            'Carrot': 'high',
            'Cucumber': 'high',
            'Watermelon': 'medium',
            'Potato': 'medium',
            'Sweet Potato': 'medium',
            'Onion': 'low',
            'Garlic': 'low',
            'Ginger': 'low',
        }
        
        self.ethylene_absorbers = ['Apple', 'Banana', 'Tomato']
        
    def load_model(self, model_path: str) -> bool:
        """
        Loads updated ethylene configurations/rules from a file.
        For a rule-based system, this might involve loading a JSON or pickle
        file containing updated self.ethylene_producers, self.ethylene_sensitive, etc.
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"EthyleneMonitor config file not found: {model_path}. Using default rules.")
                return False
            
            with open(model_path, 'rb') as f:
                loaded_data = joblib.load(f) # Using joblib for consistency
            
            if 'ethylene_producers' in loaded_data:
                self.ethylene_producers.update(loaded_data['ethylene_producers'])
            if 'ethylene_sensitive' in loaded_data:
                self.ethylene_sensitive.update(loaded_data['ethylene_sensitive'])
            # ... update other configurations
            logger.info(f"EthyleneMonitor loaded custom configuration from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading EthyleneMonitor configuration from {model_path}: {e}")
            return False

    def check_compatibility(self, fruit1, fruit2, distance_cm=50):
        """Check if two fruits can be stored together"""
        fruit1 = fruit1.capitalize()
        fruit2 = fruit2.capitalize()
        
        # Same fruit is always compatible
        if fruit1 == fruit2:
            return True, "Same fruit type - compatible"
        
        producer1_info = self.ethylene_producers.get(fruit1)
        producer2_info = self.ethylene_producers.get(fruit2)
        
        producer1_level = producer1_info['level'] if producer1_info else None
        producer2_level = producer2_info['level'] if producer2_info else None

        sensitive1 = self.ethylene_sensitive.get(fruit1)
        sensitive2 = self.ethylene_sensitive.get(fruit2)
        
        # Check for incompatibility
        if producer1_level and sensitive2:
            severity = self._get_incompatibility_severity(producer1_level, sensitive2, distance_cm)
            return False, f"{fruit1} (ethylene producer) incompatible with {fruit2} (ethylene sensitive) - Severity: {severity}"
        
        if producer2_level and sensitive1:
            severity = self._get_incompatibility_severity(producer2_level, sensitive1, distance_cm)
            return False, f"{fruit2} (ethylene producer) incompatible with {fruit1} (ethylene sensitive) - Severity: {severity}"
        
        # Check if both are high producers
        if producer1_level == 'high' and producer2_level == 'high':
            return True, "Both are ethylene producers - store in ventilated area"
        
        return True, "Compatible for storage"
    
    def _get_incompatibility_severity(self, producer_level, sensitive_level, distance_cm):
        """Calculate incompatibility severity"""
        severity_map = {
            ('high', 'very_high'): 'Critical',
            ('high', 'high'): 'High',
            ('medium', 'very_high'): 'High',
            ('medium', 'high'): 'Medium',
            ('low', 'very_high'): 'Medium',
            ('low', 'high'): 'Low',
        }
        
        base_severity = severity_map.get((producer_level, sensitive_level), 'Medium')
        
        # Adjust for distance
        if distance_cm < 10:
            return f"{base_severity} (Close proximity)"
        elif distance_cm < 50:
            return base_severity
        else:
            return f"{base_severity} (Safe distance)"
    
    def get_ethylene_management_tips(self, fruit_type):
        """Get tips for ethylene management for specific fruit"""
        fruit_type = fruit_type.capitalize()
        tips = []
        
        producer_info = self.ethylene_producers.get(fruit_type)
        if producer_info:
            level = producer_info['level']
            tips.append(f"{fruit_type} is a {level} ethylene producer")
            tips.append("Store separately from ethylene-sensitive produce")
            tips.append("Use ethylene absorbers in storage containers")
            tips.append("Maintain good ventilation in storage area")
            
            if fruit_type in self.ethylene_absorbers:
                tips.append(f"Note: {fruit_type} also absorbs ethylene - can help regulate other fruits")
        
        if fruit_type in self.ethylene_sensitive:
            level = self.ethylene_sensitive[fruit_type]
            tips.append(f"{fruit_type} is {level} ethylene-sensitive")
            tips.append("Keep away from ethylene producers")
            tips.append("Store in separate compartments or containers")
            tips.append("Monitor for premature ripening or yellowing")
            tips.append("Use ethylene-blocking packaging if available")
        
        # Add general tips
        if not tips:
            tips.append(f"No specific ethylene information for {fruit_type}")
            tips.append("Generally store in cool, dry, well-ventilated area")
            tips.append("Monitor regularly for quality changes")
        
        tips.append("Regularly check storage conditions (temperature, humidity)")
        
        return tips
    
    def predict_ethylene_risk(self, fruit_type: str, ethylene_level: float, 
                              producers_in_area: List[str], volume_m3: float = 100.0, 
                              ventilation_rate: float = 1.0) -> Dict[str, Any]:
        """
        Predicts ethylene accumulation risk and provides recommendations.
        This acts as the primary prediction method for the EthyleneMonitor.
        """
        fruit_type = fruit_type.capitalize()
        
        # Calculate background accumulation from other producers in the area
        total_production = 0
        for producer_fruit in producers_in_area:
            producer_fruit_cap = producer_fruit.capitalize()
            producer_info = self.ethylene_producers.get(producer_fruit_cap)
            if producer_info:
                total_production += producer_info['production_rate']
        
        # Ethylene concentration in ppm (simplified, assumes weight)
        # We can use the directly measured ethylene_level for immediate risk assessment
        
        concentration = ethylene_level # Using measured level as primary input
        
        # If no direct measurement, estimate from producers and accumulation
        if concentration <= 0 and total_production > 0:
             # Very simplified accumulation logic, better to have a sensor reading
            concentration = (total_production / volume_m3) / ventilation_rate 
            concentration = round(concentration, 2)
            if concentration > 50: # Cap estimated concentration
                concentration = 50.0

        risk_level = 'Low'
        confidence = 0.9
        recommendations = []

        # Determine risk and recommendations
        if concentration > 100: # Very high risk
            risk_level = 'Critical'
            confidence = 0.95
            recommendations.append('IMMEDIATE: High ethylene concentration detected. Initiate maximum ventilation.')
            recommendations.append('Remove known ethylene producers or relocate sensitive produce.')
        elif concentration > 50: # High risk
            risk_level = 'High'
            confidence = 0.9
            recommendations.append('High ethylene concentration detected. Increase ventilation and monitor closely.')
            recommendations.append('Inspect produce for accelerated ripening or spoilage.')
        elif concentration > 20: # Medium risk
            risk_level = 'Medium'
            confidence = 0.8
            recommendations.append('Elevated ethylene levels. Ensure adequate ventilation. Separate sensitive produce.')
        else:
            recommendations.append('Ethylene levels are within acceptable range.')
        
        # Fruit-specific sensitivity
        if fruit_type in self.ethylene_sensitive:
            sensitive_level = self.ethylene_sensitive[fruit_type]
            if sensitive_level == 'very_high' and concentration > 5:
                risk_level = 'High' if risk_level == 'Low' else risk_level # Upgrade risk if sensitive
                recommendations.append(f'WARNING: {fruit_type} is very ethylene-sensitive. Even moderate levels can cause issues.')
            elif sensitive_level == 'high' and concentration > 10:
                risk_level = 'Medium' if risk_level == 'Low' else risk_level
                recommendations.append(f'WARNING: {fruit_type} is ethylene-sensitive. Monitor for premature ripening.')
        
        # Add general tips
        recommendations.extend(self.get_ethylene_management_tips(fruit_type))
        
        return {
            'type': 'ethylene',
            'predicted_value': f"{risk_level} Risk ({concentration} ppm)",
            'risk_level': risk_level,
            'ethylene_concentration_ppm': concentration,
            'confidence': confidence,
            'input_conditions': {
                'fruit_type': fruit_type,
                'ethylene_level_measured': ethylene_level,
                'producers_in_area': producers_in_area,
                'volume_m3': volume_m3,
                'ventilation_rate': ventilation_rate
            },
            'recommendations': list(set(recommendations)) # Remove duplicates
        }


class FruitDiseasePredictor:
    """Predict common fruit diseases based on conditions"""
    
    def __init__(self):
        self.disease_models = self._load_disease_models()
    
    def _load_disease_models(self):
        """Load disease prediction models for common fruits"""
        return {
            'Banana': {
                'diseases': [
                    {
                        'name': 'Anthracnose',
                        'symptoms': ['Dark spots', 'Fruit rot'],
                        'conditions': {'humidity': '>90%', 'temperature': '25-30°C'},
                        'prevention': ['Proper sanitation', 'Fungicide application'],
                        'treatment': ['Remove infected fruits', 'Apply copper fungicide']
                    },
                    {
                        'name': 'Crown Rot',
                        'symptoms': ['Blackening of crown', 'Soft rot'],
                        'conditions': {'humidity': '>85%', 'temperature': '20-25°C'},
                        'prevention': ['Proper handling', 'Good air circulation'],
                        'treatment': ['Trim affected areas', 'Apply fungicide']
                    }
                ]
            },
            'Tomato': {
                'diseases': [
                    {
                        'name': 'Early Blight',
                        'symptoms': ['Concentric rings', 'Yellow leaves'],
                        'conditions': {'humidity': '>80%', 'temperature': '24-29°C'},
                        'prevention': ['Crop rotation', 'Proper spacing'],
                        'treatment': ['Apply fungicide', 'Remove infected leaves']
                    },
                    {
                        'name': 'Blossom End Rot',
                        'symptoms': ['Dark spots at blossom end'],
                        'conditions': {'calcium_deficiency': True, 'irregular_watering': True},
                        'prevention': ['Consistent watering', 'Calcium supplements'],
                        'treatment': ['Increase calcium', 'Regular watering']
                    }
                ]
            },
            'Apple': {
                'diseases': [
                    {
                        'name': 'Apple Scab',
                        'symptoms': ['Olive green spots', 'Corky lesions'],
                        'conditions': {'humidity': '>90%', 'temperature': '17-24°C'},
                        'prevention': ['Remove fallen leaves', 'Fungicide sprays'],
                        'treatment': ['Apply sulfur fungicide', 'Prune affected branches']
                    }
                ]
            }
        }
    
    def predict_disease_risk(self, fruit_type, temperature, humidity, days_in_storage):
        """Predict disease risk based on conditions"""
        fruit_type = fruit_type.capitalize()
        
        if fruit_type not in self.disease_models:
            return {
                'risk_level': 'Unknown',
                'message': f'No disease data available for {fruit_type}',
                'recommendations': ['Monitor regularly', 'Maintain optimal storage conditions']
            }
        
        diseases = self.disease_models[fruit_type]['diseases']
        risks = []
        
        for disease in diseases:
            risk_score = 0
            
            # Temperature check
            temp_range = disease['conditions'].get('temperature', '')
            if '25-30' in temp_range and 25 <= temperature <= 30:
                risk_score += 30
            elif '20-25' in temp_range and 20 <= temperature <= 25:
                risk_score += 30
            elif '17-24' in temp_range and 17 <= temperature <= 24:
                risk_score += 30
            
            # Humidity check
            humidity_req = disease['conditions'].get('humidity', '')
            if '>90' in humidity_req and humidity > 90:
                risk_score += 40
            elif '>85' in humidity_req and humidity > 85:
                risk_score += 40
            elif '>80' in humidity_req and humidity > 80:
                risk_score += 40
            
            # Storage time effect
            if days_in_storage > 7:
                risk_score += 20
            elif days_in_storage > 14:
                risk_score += 30
            
            if risk_score > 50:
                risks.append({
                    'disease': disease['name'],
                    'risk_score': risk_score,
                    'risk_level': 'High' if risk_score > 70 else 'Medium',
                    'symptoms': disease['symptoms'],
                    'prevention': disease['prevention'],
                    'treatment': disease['treatment']
                })
        
        if risks:
            highest_risk = max(risks, key=lambda x: x['risk_score'])
            return {
                'risk_level': highest_risk['risk_level'],
                'diseases_at_risk': risks,
                'highest_risk_disease': highest_risk['disease'],
                'recommendations': highest_risk['prevention']
            }
        else:
            return {
                'risk_level': 'Low',
                'message': 'No significant disease risk detected',
                'recommendations': ['Continue monitoring', 'Maintain current storage conditions']
            }


class FruitPricePredictor:
    """Predict fruit prices based on quality and market factors"""
    
    def __init__(self):
        self.base_prices = {
            'Banana': {'Fresh': 2000, 'Good': 1500, 'Fair': 1000, 'Poor': 500, 'Rotten': 100},
            'Apple': {'Fresh': 3000, 'Good': 2500, 'Fair': 1800, 'Poor': 800, 'Rotten': 200},
            'Orange': {'Fresh': 2500, 'Good': 2000, 'Fair': 1500, 'Poor': 700, 'Rotten': 150},
            'Mango': {'Fresh': 3500, 'Good': 2800, 'Fair': 2000, 'Poor': 1000, 'Rotten': 300},
            'Tomato': {'Fresh': 1500, 'Good': 1200, 'Fair': 800, 'Poor': 400, 'Rotten': 100},
        }
        
        self.market_factors = {
            'seasonality': {
                'Banana': {'high': 1.2, 'low': 0.8, 'peak_months': [3, 4, 5]},
                'Apple': {'high': 1.3, 'low': 0.7, 'peak_months': [9, 10, 11]},
                'Mango': {'high': 1.5, 'low': 0.5, 'peak_months': [1, 2, 3]},
            },
            'demand': {
                'holiday': 1.3,
                'weekend': 1.1,
                'normal': 1.0
            }
        }
    
    def predict_price(self, fruit_type, quality, quantity_kg=1, market_conditions='normal'):
        """Predict price based on quality and market factors"""
        fruit_type = fruit_type.capitalize()
        quality = quality.capitalize()
        
        # Get base price
        base_price = self.base_prices.get(fruit_type, {}).get(quality, 1000)
        
        # Apply quality multiplier
        quality_multiplier = {
            'Fresh': 1.0,
            'Good': 0.8,
            'Fair': 0.6,
            'Poor': 0.3,
            'Rotten': 0.1
        }.get(quality, 0.5)
        
        # Apply seasonality
        import datetime
        current_month = datetime.datetime.now().month
        seasonality = 1.0
        
        if fruit_type in self.market_factors['seasonality']:
            season_data = self.market_factors['seasonality'][fruit_type]
            if current_month in season_data['peak_months']:
                seasonality = season_data['high']
            else:
                seasonality = season_data['low']
        
        # Apply demand factor
        demand_factor = self.market_factors['demand'].get(market_conditions, 1.0)
        
        # Quantity discount
        quantity_factor = 1.0
        if quantity_kg > 100:
            quantity_factor = 0.9
        elif quantity_kg > 50:
            quantity_factor = 0.95
        
        # Calculate final price per kg
        price_per_kg = base_price * quality_multiplier * seasonality * demand_factor * quantity_factor
        
        # Calculate total price
        total_price = price_per_kg * quantity_kg
        
        return {
            'price_per_kg': round(price_per_kg, 2),
            'total_price': round(total_price, 2),
            'base_price': base_price,
            'factors': {
                'quality_multiplier': quality_multiplier,
                'seasonality_factor': seasonality,
                'demand_factor': demand_factor,
                'quantity_factor': quantity_factor
            },
            'recommendations': self._get_pricing_recommendations(
                fruit_type, quality, price_per_kg, seasonality
            )
        }
    
    def _get_pricing_recommendations(self, fruit_type, quality, current_price, seasonality):
        """Get pricing recommendations"""
        recommendations = []
        
        if quality in ['Poor', 'Rotten']:
            recommendations.append("Consider processing into value-added products")
            recommendations.append("Offer at deep discount for immediate sale")
            recommendations.append("Use for animal feed or composting")
        elif quality == 'Fair':
            recommendations.append("Market as 'economy' or 'bargain' quality")
            recommendations.append("Bundle with other products")
            recommendations.append("Target price-sensitive customers")
        elif quality in ['Fresh', 'Good']:
            if seasonality > 1.2:
                recommendations.append("Premium pricing justified during peak season")
                recommendations.append("Focus on quality presentation")
                recommendations.append("Target premium markets and restaurants")
            else:
                recommendations.append("Competitive pricing recommended")
                recommendations.append("Highlight freshness and quality")
                recommendations.append("Consider value bundles")
        
        # General recommendations
        recommendations.append("Monitor competitor prices regularly")
        recommendations.append("Adjust pricing based on daily market conditions")
        recommendations.append("Consider dynamic pricing based on remaining shelf life")
        
        return recommendations


# ==================== MAIN AI SERVICE ====================

class ShelfLifePredictor:
    """Predicts remaining shelf life and optimal storage conditions for fruits."""

    def __init__(self):
        # Base shelf life (days) for various fruits under optimal conditions
        # This can be made configurable or loaded from a database
        self.base_shelf_lives = {
            'Banana': {'min': 7, 'max': 14, 'optimal_temp': 13, 'optimal_humidity': 90, 'ethylene_sensitive': True, 'climacteric': True},
            'Apple': {'min': 30, 'max': 180, 'optimal_temp': 1, 'optimal_humidity': 90, 'ethylene_sensitive': False, 'climacteric': True},
            'Orange': {'min': 21, 'max': 45, 'optimal_temp': 5, 'optimal_humidity': 90, 'ethylene_sensitive': False, 'climacteric': False},
            'Mango': {'min': 7, 'max': 21, 'optimal_temp': 13, 'optimal_humidity': 90, 'ethylene_sensitive': True, 'climacteric': True},
            'Tomato': {'min': 7, 'max': 14, 'optimal_temp': 13, 'optimal_humidity': 90, 'ethylene_sensitive': True, 'climacteric': True},
            'Strawberry': {'min': 3, 'max': 7, 'optimal_temp': 0, 'optimal_humidity': 95, 'ethylene_sensitive': False, 'climacteric': False},
            'Avocado': {'min': 4, 'max': 10, 'optimal_temp': 7, 'optimal_humidity': 90, 'ethylene_sensitive': True, 'climacteric': True},
        }

        # Temperature effects: multiplier on shelf life (lower temp, higher multiplier up to a point)
        self.temp_factors = {
            'optimal': lambda temp, optimal: 1.0 if abs(temp - optimal) <= 2 else (0.8 if abs(temp - optimal) <= 5 else 0.5),
            'very_low': 0.3, # chilling injury risk
            'high': 0.6, # accelerated decay
            'very_high': 0.2,
        }

        # Humidity effects: multiplier on shelf life
        self.humidity_factors = {
            'optimal': lambda hum: 1.0 if 85 <= hum <= 95 else (0.8 if 70 <= hum <= 85 or 95 < hum <= 98 else 0.6),
            'low': 0.5, # dehydration
            'very_low': 0.3,
            'high': 0.7, # mold risk
            'very_high': 0.4,
        }

        # Ethylene effects: multiplier on shelf life for climacteric fruits
        self.ethylene_factors = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.5,
            'very_high': 0.3,
        }

        # CO2 effects: multiplier on shelf life (high CO2 can extend for some)
        self.co2_factors = {
            'low': 1.0,
            'medium': 1.1, # beneficial for some fruits
            'high': 1.2, # CA storage levels
        }

    def load_model(self, model_path: str) -> bool:
        """
        Loads a pre-trained shelf life model or configuration from a file.
        For this rule-based predictor, it might load updated rules or base_shelf_lives.
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Shelf life model file not found: {model_path}. Using default rules.")
                return False
            
            # Example: Load updated base_shelf_lives or factors from a JSON/pickle file
            with open(model_path, 'rb') as f:
                loaded_data = joblib.load(f) # Using joblib for consistency
            
            if 'base_shelf_lives' in loaded_data:
                self.base_shelf_lives.update(loaded_data['base_shelf_lives'])
            if 'temp_factors' in loaded_data:
                self.temp_factors.update(loaded_data['temp_factors'])
            # ... update other factors as needed
            logger.info(f"ShelfLifePredictor loaded custom configuration from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading ShelfLifePredictor configuration from {model_path}: {e}")
            return False

    def predict_shelf_life(self, product: Any, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts the remaining shelf life of a product based on its type, age,
        and current environmental conditions.
        """
        fruit_type = product.fruit_type.name if hasattr(product.fruit_type, 'name') else str(product.fruit_type)
        fruit_type = fruit_type.capitalize()

        base_info = self.base_shelf_lives.get(fruit_type)
        if not base_info:
            return {
                'predicted_days_remaining': 0,
                'predicted_expiry_date': None,
                'status': 'Unknown',
                'confidence': 0.0,
                'recommendations': [f"No shelf life data for {fruit_type}. Add to configuration."]
            }

        # Get relevant data points
        current_temp = sensor_data.get('temperature', base_info['optimal_temp'])
        current_humidity = sensor_data.get('humidity', base_info['optimal_humidity'])
        ethylene_level = sensor_data.get('ethylene_level', 0.0) # ppm
        co2_level = sensor_data.get('co2_level', 400.0) # ppm
        
        harvest_date = getattr(product, 'harvest_date', None)
        production_date = getattr(product, 'production_date', None)

        if not harvest_date and not production_date:
            # Fallback for products without a specific start date
            age_in_days = 0 
            logger.warning(f"No harvest_date or production_date for product {product.id}. Assuming fresh.")
        else:
            start_date = harvest_date if harvest_date else production_date
            age_in_days = (timezone.now().date() - start_date).days

        # Start with max base shelf life days
        estimated_shelf_life_days = base_info['max']

        # Apply factors based on current conditions
        # Temperature Factor
        temp_factor_val = 1.0
        if current_temp < (base_info['optimal_temp'] - 5): # Very low temp, chilling injury risk
            temp_factor_val = self.temp_factors['very_low']
        elif current_temp > (base_info['optimal_temp'] + 5): # High temp, accelerated decay
            temp_factor_val = self.temp_factors['high']
        elif current_temp > (base_info['optimal_temp'] + 10): # Very high temp
            temp_factor_val = self.temp_factors['very_high']
        else: # Near optimal
            temp_factor_val = self.temp_factors['optimal'](current_temp, base_info['optimal_temp'])
        estimated_shelf_life_days *= temp_factor_val

        # Humidity Factor
        hum_factor_val = self.humidity_factors['optimal'](current_humidity)
        estimated_shelf_life_days *= hum_factor_val
        
        # Ethylene Factor (only for climacteric and sensitive fruits)
        if base_info['climacteric'] and base_info['ethylene_sensitive']:
            if ethylene_level > 10: # High ethylene
                estimated_shelf_life_days *= self.ethylene_factors['high']
            elif ethylene_level > 1: # Medium ethylene
                estimated_shelf_life_days *= self.ethylene_factors['medium']
        
        # CO2 Factor (simplified for general benefits or detriments)
        if co2_level > 5000: # High CO2, potentially beneficial for CA storage
             estimated_shelf_life_days *= self.co2_factors['high']
        elif co2_level > 1000:
             estimated_shelf_life_days *= self.co2_factors['medium']

        # Reduce by age
        predicted_days_remaining = max(0, int(estimated_shelf_life_days) - age_in_days)
        predicted_expiry_date = timezone.now().date() + timedelta(days=predicted_days_remaining)

        # Determine status and confidence
        status, confidence, recommendations = self._determine_status_and_recommendations(
            predicted_days_remaining, current_temp, current_humidity, ethylene_level, base_info
        )

        return {
            'type': 'shelf_life',
            'predicted_value': f"{predicted_days_remaining} days",
            'predicted_days_remaining': predicted_days_remaining,
            'predicted_expiry_date': predicted_expiry_date.isoformat(),
            'status': status,
            'confidence': confidence,
            'input_conditions': {
                'fruit_type': fruit_type,
                'age_in_days': age_in_days,
                'temperature': current_temp,
                'humidity': current_humidity,
                'ethylene_level': ethylene_level,
                'co2_level': co2_level,
            },
            'recommendations': recommendations
        }

    def _determine_status_and_recommendations(self, days_remaining: int, temp: float, hum: float, ethylene: float, base_info: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Determines the status and generates recommendations for shelf life."""
        recommendations = []
        status = 'Optimal'
        confidence = 0.9 # Base confidence

        if days_remaining <= 2:
            status = 'Critical'
            confidence *= 0.7
            recommendations.append("URGENT: Product shelf life is critical. Prioritize for immediate sale or processing.")
            recommendations.append("Verify actual product quality for potential disposal.")
        elif days_remaining <= 5:
            status = 'Warning'
            confidence *= 0.8
            recommendations.append("WARNING: Product nearing end of shelf life. Plan for quick turnover or discount.")
        else:
            recommendations.append("Product has good remaining shelf life.")

        # Environmental condition recommendations (enhance beyond simple alerts)
        if temp < (base_info['optimal_temp'] - 2):
            recommendations.append(f"Temperature is below optimal ({base_info['optimal_temp']}°C). Adjust to avoid chilling injury.")
            confidence *= 0.9
        elif temp > (base_info['optimal_temp'] + 2):
            recommendations.append(f"Temperature is above optimal ({base_info['optimal_temp']}°C). Reduce to slow ripening/decay.")
            confidence *= 0.9

        if hum < (base_info['optimal_humidity'] - 5):
            recommendations.append(f"Humidity is low ({base_info['optimal_humidity']}% optimal). Increase to prevent dehydration.")
            confidence *= 0.9
        elif hum > (base_info['optimal_humidity'] + 5):
            recommendations.append(f"Humidity is high ({base_info['optimal_humidity']}% optimal). Reduce to prevent mold/bacterial growth.")
            confidence *= 0.9

        if base_info['climacteric'] and base_info['ethylene_sensitive'] and ethylene > 0.5:
            recommendations.append("Ethylene levels are elevated. Isolate from ethylene-producing fruits or use ethylene scrubbers.")
            confidence *= 0.8

        return status, min(1.0, confidence), list(set(recommendations)) # Remove duplicates

class BikaAIService:
    """Main AI service orchestrating all prediction models"""
    
    def __init__(self):
        self.quality_predictor = FruitQualityPredictor()
        self.ripeness_predictor = FruitRipenessPredictor()
        self.ethylene_monitor = EthyleneMonitor()
        self.disease_predictor = FruitDiseasePredictor()
        self.price_predictor = FruitPricePredictor()
        
        # We no longer call load_pre_trained_models here as EnhancedBikaAIService handles comprehensive loading.
    
    def load_pre_trained_models(self):
        """Load pre-trained models from disk"""
        model_dir = os.path.join(settings.MEDIA_ROOT, 'fruit_models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Look for pre-trained models
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if model_files:
            # Load the most recent model
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
            
            try:
                if self.quality_predictor.load_model(model_path):
                    print(f"Loaded pre-trained model: {latest_model}")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
    
    def train_fruit_quality_model(self, csv_file, model_type='auto'):
        """Train fruit quality prediction model from CSV"""
        try:
            # Save uploaded file
            timestamp = int(timezone.now().timestamp())
            file_path = os.path.join('fruit_datasets', f'dataset_{timestamp}.csv')
            saved_path = default_storage.save(file_path, csv_file)
            full_path = default_storage.path(saved_path)
            
            # Set model type
            if model_type == 'auto':
                self.quality_predictor.model_type = 'random_forest'
            else:
                self.quality_predictor.model_type = model_type
            
            # Load and train
            X, y, df = self.quality_predictor.load_fruit_dataset(full_path)
            
            if X is None or y is None:
                return {'success': False, 'error': 'Failed to load or validate CSV data'}
            
            # Train model
            results = self.quality_predictor.train_model(X, y)
            
            if 'error' in results:
                return {'success': False, 'error': results['error']}
            
            # Save model
            model_filename = f'fruit_quality_model_{self.quality_predictor.model_type}_{timestamp}.pkl'
            model_dir = os.path.join(settings.MEDIA_ROOT, 'fruit_models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_filename)
            
            self.quality_predictor.save_model(model_path)
            
            # Extract insights from dataset
            dataset_insights = {
                'total_samples': len(df),
                'fruit_distribution': df['fruit_type'].value_counts().to_dict(),
                'quality_distribution': df['quality_class'].value_counts().to_dict(),
                'avg_temperature': df['temperature'].mean(),
                'avg_humidity': df['humidity'].mean(),
                'data_quality_score': self._calculate_data_quality(df)
            }
            
            return {
                'success': True,
                'model_metrics': results,
                'model_path': model_path,
                'model_type': self.quality_predictor.model_type,
                'dataset_insights': dataset_insights,
                'training_samples': results['training_samples'],
                'test_accuracy': results['accuracy'],
                'cross_val_mean': results.get('cv_mean', 0),
                'unique_fruits': df['fruit_type'].unique().tolist()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_data_quality(self, df):
        """Calculate data quality score (0-100)"""
        score = 100
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        score -= missing_pct * 2
        
        # Check for duplicates
        duplicate_pct = df.duplicated().sum() / len(df) * 100
        score -= duplicate_pct
        
        # Check data distribution
        for col in ['temperature', 'humidity', 'light_intensity', 'co2_level']:
            if col in df.columns:
                # Check for outliers using IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                outlier_pct = len(outliers) / len(df) * 100
                score -= outlier_pct / 2
        
        return max(0, min(100, score))
    
    def predict_fruit_quality(self, fruit_name, temperature, humidity, 
                            light_intensity, co2_level, batch_id=None):
        """Comprehensive fruit quality prediction"""
        try:
            # Quality prediction
            quality_prediction = self.quality_predictor.predict_quality(
                fruit_name, temperature, humidity, light_intensity, co2_level
            )
            
            # Ripeness prediction
            ripeness_prediction = self.ripeness_predictor.predict_ripeness(
                fruit_name, temperature, 0, 3, humidity, light_intensity
            )
            
            # Disease risk assessment
            disease_risk = self.disease_predictor.predict_disease_risk(
                fruit_name, temperature, humidity, 5
            )
            
            # Price prediction
            price_prediction = self.price_predictor.predict_price(
                fruit_name, quality_prediction['predicted_class']
            )
            
            # Storage recommendations
            storage_recommendations = self._generate_storage_recommendations(
                fruit_name, quality_prediction['predicted_class'],
                temperature, humidity, light_intensity, co2_level
            )
            
            # Combine all predictions
            comprehensive_result = {
                'success': True,
                'quality_prediction': quality_prediction,
                'ripeness_prediction': ripeness_prediction,
                'disease_risk': disease_risk,
                'price_prediction': price_prediction,
                'storage_recommendations': storage_recommendations,
                'ethylene_tips': self.ethylene_monitor.get_ethylene_management_tips(fruit_name),
                'timestamp': timezone.now().isoformat(),
                'batch_id': batch_id
            }
            
            return comprehensive_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_storage_recommendations(self, fruit_name, quality_class, 
                                         temp, humidity, light, co2):
        """Generate storage recommendations"""
        recommendations = []
        
        # Temperature recommendations
        if temp < 2:
            recommendations.append("WARNING: Temperature too low - risk of chilling injury")
        elif temp > 12:
            recommendations.append("WARNING: Temperature too high - accelerated spoilage")
        
        # Humidity recommendations
        if humidity < 85:
            recommendations.append("Increase humidity to prevent dehydration")
        elif humidity > 95:
            recommendations.append("Reduce humidity to prevent mold growth")
        
        # Light recommendations
        if light > 100:
            recommendations.append("Reduce light exposure to prevent photo-degradation")
        
        # CO₂ recommendations
        if co2 > 1000:
            recommendations.append("Improve ventilation - high CO₂ levels detected")
        
        # Quality-specific recommendations
        if quality_class in ['Poor', 'Rotten']:
            recommendations.append("URGENT: Consider immediate processing or disposal")
            recommendations.append("Isolate from other fruits to prevent contamination")
        
        # Fruit-specific recommendations
        fruit_specific = {
            'Banana': ['Store at 13-15°C', 'Keep away from apples and tomatoes'],
            'Apple': ['Store at 0-4°C', 'Can be stored for long periods'],
            'Mango': ['Store at 10-13°C', 'Handle carefully to avoid bruising'],
            'Tomato': ['Never refrigerate', 'Store at 10-15°C'],
        }
        
        recommendations.extend(fruit_specific.get(fruit_name, []))
        
        return recommendations
    
    def analyze_batch_trends(self, batch_id, days=7):
        """Analyze trends for a fruit batch"""
        try:
            from bika.models import FruitBatch, FruitQualityReading
            
            batch = FruitBatch.objects.get(id=batch_id)
            readings = FruitQualityReading.objects.filter(
                fruit_batch=batch,
                timestamp__gte=timezone.now() - timedelta(days=days)
            ).order_by('timestamp')
            
            if not readings.exists():
                return {'error': 'No readings available for analysis'}
            
            # Convert to DataFrame for analysis
            data = []
            for reading in readings:
                data.append({
                    'timestamp': reading.timestamp,
                    'temperature': float(reading.temperature),
                    'humidity': float(reading.humidity),
                    'predicted_class': reading.predicted_class,
                    'confidence': float(reading.confidence_score)
                })
            
            df = pd.DataFrame(data)
            
            # Calculate trends
            temp_trend = self._calculate_trend(df['temperature'])
            humidity_trend = self._calculate_trend(df['humidity'])
            
            # Quality trend
            quality_scores = {'Fresh': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Rotten': 1}
            df['quality_score'] = df['predicted_class'].map(quality_scores)
            quality_trend = self._calculate_trend(df['quality_score'])
            
            # Predict future quality
            future_predictions = self._predict_future_quality(df, days_ahead=3)
            
            return {
                'success': True,
                'batch_info': {
                    'batch_number': batch.batch_number,
                    'fruit_type': batch.fruit_type.name,
                    'days_remaining': batch.days_remaining
                },
                'statistics': {
                    'readings_count': len(df),
                    'avg_temperature': df['temperature'].mean(),
                    'avg_humidity': df['humidity'].mean(),
                    'avg_confidence': df['confidence'].mean(),
                    'current_quality': df.iloc[-1]['predicted_class'] if len(df) > 0 else 'Unknown'
                },
                'trends': {
                    'temperature': temp_trend,
                    'humidity': humidity_trend,
                    'quality': quality_trend
                },
                'future_predictions': future_predictions,
                'recommendations': self._generate_batch_recommendations(df, batch)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_trend(self, series):
        """Calculate trend (increasing, decreasing, stable)"""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _predict_future_quality(self, df, days_ahead=3):
        """Predict future quality based on trends"""
        if len(df) < 5:
            return {'error': 'Insufficient data for prediction'}
        
        try:
            # Simple moving average prediction
            quality_scores = df['quality_score'].values
            window = min(3, len(quality_scores))
            
            # Predict using moving average
            predictions = []
            for i in range(days_ahead):
                if len(quality_scores) >= window:
                    last_values = quality_scores[-window:]
                    predicted = np.mean(last_values)
                    predictions.append(predicted)
                    quality_scores = np.append(quality_scores, predicted)
            
            # Convert scores back to quality classes
            score_to_class = {5: 'Fresh', 4: 'Good', 3: 'Fair', 2: 'Poor', 1: 'Rotten'}
            predicted_classes = []
            for score in predictions:
                # Find closest score
                closest = min(score_to_class.keys(), key=lambda x: abs(x - score))
                predicted_classes.append(score_to_class[closest])
            
            return {
                'predicted_quality': predicted_classes,
                'predicted_scores': [float(score) for score in predictions],
                'confidence': 0.7  # Simplified confidence score
            }
            
        except Exception:
            return {'error': 'Prediction failed'}
    
    def _generate_batch_recommendations(self, df, batch):
        """Generate recommendations for a batch"""
        recommendations = []
        
        # Check temperature stability
        temp_std = df['temperature'].std()
        if temp_std > 2:
            recommendations.append("Temperature fluctuations detected - stabilize storage conditions")
        
        # Check humidity
        avg_humidity = df['humidity'].mean()
        if avg_humidity < 85:
            recommendations.append("Increase humidity to optimal range")
        elif avg_humidity > 95:
            recommendations.append("Reduce humidity to prevent mold")
        
        # Check quality trend
        quality_scores = df['quality_score'].values
        if len(quality_scores) >= 2:
            if quality_scores[-1] < quality_scores[0] - 1:
                recommendations.append("Quality deterioration detected - consider priority sale")
        
        # Days remaining
        if batch.days_remaining <= 2:
            recommendations.append("URGENT: Batch approaching expiry - immediate action required")
        elif batch.days_remaining <= 5:
            recommendations.append("Batch nearing expiry - plan for sale or processing")
        
        return recommendations


# ==================== GLOBAL INSTANCE ====================

# Create global AI service instance
bika_ai_service = BikaAIService()

# Export classes and instances
__all__ = [
    'FruitQualityPredictor',
    'FruitRipenessPredictor',
    'EthyleneMonitor',
    'FruitDiseasePredictor',
    'FruitPricePredictor',
    'ShelfLifePredictor', # Added
    'BikaAIService',
    'bika_ai_service'
]