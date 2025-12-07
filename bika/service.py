# bika/service.py - ALL AI SERVICES IN ONE FILE
import os
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

warnings.filterwarnings('ignore')

# ==================== IMPORT DEPENDENCIES WITH FALLBACKS ====================

# Try to import scikit-learn
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Some AI features will be disabled.")

# Try to import joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("joblib not available. Model saving/loading will not work.")

# Try to import xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available.")

# Try to import tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Neural networks disabled.")

# ==================== AI MODELS ====================

class FruitQualityPredictor:
    """AI model for predicting fruit quality based on environmental conditions"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.class_names = ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
    
    def load_fruit_dataset(self, csv_path):
        """Load and prepare fruit quality dataset"""
        if not SKLEARN_AVAILABLE:
            return None, None, None
            
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
                ('cat', categorical_transformer, ['Fruit'])
            ]
        )
        
        # Fit preprocessor
        self.preprocessor.fit(X)
    
    def train_model(self, X, y, test_size=0.2, cv_folds=5):
        """Train the selected model with cross-validation"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
            
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
                class_weight='balanced'
            )
            
        elif self.model_type == 'xgboost':
            if XGBOOST_AVAILABLE:
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
            else:
                print("XGBoost not available. Using Random Forest instead.")
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
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
            if TENSORFLOW_AVAILABLE:
                self.model = self._create_neural_network(X_train_processed.shape[1])
            else:
                print("TensorFlow not available. Using Random Forest instead.")
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Cross-validation
        if self.model_type != 'neural_network' and not (self.model_type == 'neural_network' and not TENSORFLOW_AVAILABLE):
            cv_scores = cross_val_score(
                self.model, X_train_processed, y_train,
                cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train model
        if self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
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
        if not TENSORFLOW_AVAILABLE:
            return None
            
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
        if self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
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
    
    def save_model(self, model_path):
        """Save trained model and preprocessor"""
        if not JOBLIB_AVAILABLE:
            print(f"Warning: joblib not available. Model not saved to {model_path}")
            return
        
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
        if not JOBLIB_AVAILABLE:
            print(f"Error: joblib not available. Cannot load model from {model_path}")
            return False
        
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
        # Simple rule-based system
        ripeness_factors = {
            'temperature': temperature,
            'ethylene': ethylene_level,
            'days': days_since_harvest
        }
        
        ripening_rates = {
            'Banana': 0.8,
            'Apple': 0.3,
            'Orange': 0.4,
            'Mango': 1.0,
            'Tomato': 0.9,
            'Avocado': 0.7
        }
        
        base_rate = ripening_rates.get(fruit_type, 0.5)
        ripeness_score = min(1.0, base_rate * (1 + ethylene_level/100) * (1 + days_since_harvest/10))
        
        if ripeness_score < 0.3:
            return 'unripe'
        elif ripeness_score < 0.7:
            return 'ripe'
        else:
            return 'overripe'
    
    def estimate_shelf_life(self, fruit_type, current_quality, temperature, humidity):
        """Estimate remaining shelf life"""
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

# ==================== AI SERVICES ====================

class FruitAIService:
    """AI Service specialized for fruit quality monitoring"""
    
    def __init__(self):
        self.quality_predictor = None
        self.ripeness_predictor = FruitRipenessPredictor()
        self.ethylene_monitor = EthyleneMonitor()
        self.loaded_models = {}
    
    def train_fruit_quality_model(self, csv_file, model_type='random_forest'):
        """Train fruit quality prediction model from CSV"""
        try:
            # Save uploaded file
            timestamp = int(timezone.now().timestamp())
            file_path = os.path.join('fruit_datasets', f'dataset_{timestamp}.csv')
            saved_path = default_storage.save(file_path, csv_file)
            full_path = default_storage.path(saved_path)
            
            # Initialize predictor
            predictor = FruitQualityPredictor(model_type=model_type)
            
            # Load and train
            X, y, df = predictor.load_fruit_dataset(full_path)
            
            if X is None or y is None:
                return {'success': False, 'error': 'Failed to load CSV data'}
            
            # Train model
            results = predictor.train_model(X, y)
            
            if 'error' in results:
                return {'success': False, 'error': results['error']}
            
            # Save model
            model_filename = f'fruit_quality_model_{model_type}_{timestamp}.pkl'
            model_dir = os.path.join(settings.MEDIA_ROOT, 'fruit_models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_filename)
            
            predictor.save_model(model_path)
            
            # Store in memory
            self.loaded_models['fruit_quality'] = predictor
            
            # Extract unique fruits from dataset
            unique_fruits = df['Fruit'].unique().tolist()
            
            return {
                'success': True,
                'accuracy': results['accuracy'],
                'model_path': model_path,
                'model_type': model_type,
                'training_samples': results['training_samples'],
                'test_accuracy': results['accuracy'],
                'unique_fruits': unique_fruits,
                'classification_report': results['classification_report']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_fruit_quality(self, fruit_name, temperature, humidity, 
                            light_intensity, co2_level, batch_id=None):
        """Predict quality for specific fruit"""
        try:
            # Load model if not loaded
            if 'fruit_quality' not in self.loaded_models:
                model_path = self._find_latest_fruit_model()
                if model_path and os.path.exists(model_path):
                    predictor = FruitQualityPredictor()
                    predictor.load_model(model_path)
                    self.loaded_models['fruit_quality'] = predictor
                else:
                    return {'error': 'No trained fruit quality model available'}
            
            predictor = self.loaded_models['fruit_quality']
            
            # Make prediction
            prediction = predictor.predict_quality(
                fruit_name, temperature, humidity, light_intensity, co2_level
            )
            
            # Get recommendations
            recommendations = self._generate_fruit_recommendations(
                fruit_name, prediction['predicted_class'], 
                temperature, humidity, light_intensity, co2_level
            )
            
            # Estimate shelf life
            estimated_shelf_life = self.ripeness_predictor.estimate_shelf_life(
                fruit_name, prediction['predicted_class'], temperature, humidity
            )
            
            # Import models here to avoid circular imports
            from bika.models import FruitBatch, FruitQualityReading, FruitType
            
            # Get fruit type info if available
            fruit_info = {}
            try:
                fruit_type = FruitType.objects.filter(name__icontains=fruit_name).first()
                if fruit_type:
                    fruit_info = {
                        'optimal_temp': f"{fruit_type.optimal_temp_min}-{fruit_type.optimal_temp_max}°C",
                        'optimal_humidity': f"{fruit_type.optimal_humidity_min}-{fruit_type.optimal_humidity_max}%",
                        'shelf_life': f"{fruit_type.shelf_life_days} days"
                    }
            except:
                pass
            
            # Save prediction to database if batch_id provided
            if batch_id:
                try:
                    batch = FruitBatch.objects.get(id=batch_id)
                    
                    FruitQualityReading.objects.create(
                        fruit_batch=batch,
                        temperature=temperature,
                        humidity=humidity,
                        light_intensity=light_intensity,
                        co2_level=co2_level,
                        predicted_class=prediction['predicted_class'],
                        confidence_score=prediction['confidence'],
                        notes=f"AI prediction: {prediction['predicted_class']}"
                    )
                except:
                    pass
            
            return {
                'success': True,
                'prediction': prediction,
                'fruit_info': fruit_info,
                'recommendations': recommendations,
                'estimated_shelf_life_days': estimated_shelf_life,
                'ethylene_tips': self.ethylene_monitor.get_ethylene_management_tips(fruit_name)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _check_optimal_conditions(self, fruit_name, temp, humidity, light, co2):
        """Check if conditions are optimal for the fruit"""
        try:
            from bika.models import FruitType
            fruit_type = FruitType.objects.filter(name__icontains=fruit_name).first()
            if not fruit_type:
                return None
            
            return (
                fruit_type.optimal_temp_min <= temp <= fruit_type.optimal_temp_max and
                fruit_type.optimal_humidity_min <= humidity <= fruit_type.optimal_humidity_max and
                light <= fruit_type.optimal_light_max and
                co2 <= fruit_type.optimal_co2_max
            )
        except:
            return None
    
    def _generate_fruit_recommendations(self, fruit_name, quality_class, 
                                      temp, humidity, light, co2):
        """Generate recommendations for fruit storage"""
        recommendations = []
        
        try:
            from bika.models import FruitType
            fruit_type = FruitType.objects.filter(name__icontains=fruit_name).first()
            if fruit_type:
                # Temperature recommendations
                if temp < fruit_type.optimal_temp_min:
                    recommendations.append(
                        f"Increase temperature to {fruit_type.optimal_temp_min}-{fruit_type.optimal_temp_max}°C"
                    )
                elif temp > fruit_type.optimal_temp_max:
                    recommendations.append(
                        f"Decrease temperature to {fruit_type.optimal_temp_min}-{fruit_type.optimal_temp_max}°C"
                    )
                
                # Humidity recommendations
                if humidity < fruit_type.optimal_humidity_min:
                    recommendations.append(
                        f"Increase humidity to {fruit_type.optimal_humidity_min}-{fruit_type.optimal_humidity_max}%"
                    )
                elif humidity > fruit_type.optimal_humidity_max:
                    recommendations.append(
                        f"Decrease humidity to {fruit_type.optimal_humidity_min}-{fruit_type.optimal_humidity_max}%"
                    )
                
                # Light recommendations
                if light > fruit_type.optimal_light_max:
                    recommendations.append(
                        f"Reduce light exposure below {fruit_type.optimal_light_max} lux"
                    )
                
                # CO₂ recommendations
                if co2 > fruit_type.optimal_co2_max:
                    recommendations.append(
                        f"Improve ventilation to reduce CO₂ below {fruit_type.optimal_co2_max} ppm"
                    )
            
            # Quality-based recommendations
            if quality_class in ['Poor', 'Rotten']:
                recommendations.append("Consider immediate sale or processing")
                recommendations.append("Check for contamination with other fruits")
            
            if fruit_type and fruit_type.chilling_sensitive and temp < 5:
                recommendations.append("Warning: Fruit is chilling sensitive - avoid temperatures below 5°C")
            
            if fruit_type and fruit_type.ethylene_sensitive:
                recommendations.append("Keep away from ethylene-producing fruits")
            
        except:
            pass
        
        return recommendations
    
    def get_batch_quality_report(self, batch_id, hours=24):
        """Generate quality report for fruit batch"""
        try:
            from bika.models import FruitBatch, FruitQualityReading, RealTimeSensorData
            
            batch = FruitBatch.objects.get(id=batch_id)
            
            # Get recent quality readings
            time_threshold = timezone.now() - timedelta(hours=hours)
            readings = FruitQualityReading.objects.filter(
                fruit_batch=batch,
                timestamp__gte=time_threshold
            ).order_by('timestamp')
            
            if not readings.exists():
                return {'error': 'No quality readings available'}
            
            # Calculate statistics
            qualities = [r.predicted_class for r in readings]
            confidences = [float(r.confidence_score) for r in readings]
            
            quality_distribution = pd.Series(qualities).value_counts().to_dict()
            
            # Calculate quality trend
            quality_scores = {
                'Fresh': 5,
                'Good': 4,
                'Fair': 3,
                'Poor': 2,
                'Rotten': 1
            }
            
            quality_trend = []
            for reading in readings:
                score = quality_scores.get(reading.predicted_class, 3)
                quality_trend.append({
                    'timestamp': reading.timestamp.isoformat(),
                    'score': score,
                    'quality': reading.predicted_class
                })
            
            # Get current conditions
            latest = readings.last()
            current_conditions = {
                'temperature': float(latest.temperature),
                'humidity': float(latest.humidity),
                'light_intensity': float(latest.light_intensity),
                'co2_level': latest.co2_level,
                'quality': latest.predicted_class,
                'confidence': float(latest.confidence_score)
            }
            
            # Calculate deterioration rate
            if len(quality_trend) > 1:
                scores = [q['score'] for q in quality_trend]
                deterioration_rate = (scores[0] - scores[-1]) / len(scores)
            else:
                deterioration_rate = 0
            
            # Predict remaining shelf life
            estimated_life = self.ripeness_predictor.estimate_shelf_life(
                batch.fruit_type.name,
                latest.predicted_class,
                latest.temperature,
                latest.humidity
            )
            
            return {
                'batch_info': {
                    'batch_number': batch.batch_number,
                    'fruit_type': batch.fruit_type.name,
                    'arrival_date': batch.arrival_date.isoformat(),
                    'expected_expiry': batch.expected_expiry.isoformat(),
                    'days_remaining': batch.days_remaining
                },
                'current_conditions': current_conditions,
                'quality_distribution': quality_distribution,
                'quality_trend': quality_trend,
                'statistics': {
                    'total_readings': len(readings),
                    'avg_confidence': np.mean(confidences) if confidences else 0,
                    'deterioration_rate': deterioration_rate,
                    'estimated_remaining_days': estimated_life
                },
                'recommendations': self._generate_fruit_recommendations(
                    batch.fruit_type.name,
                    latest.predicted_class,
                    latest.temperature,
                    latest.humidity,
                    latest.light_intensity,
                    latest.co2_level
                )
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def monitor_storage_compatibility(self, storage_location_id):
        """Check if fruits in storage are compatible"""
        try:
            from bika.models import StorageLocation, FruitBatch, RealTimeSensorData
            
            location = StorageLocation.objects.get(id=storage_location_id)
            
            # Get all fruit batches in this location
            batches = FruitBatch.objects.filter(
                storage_location=location,
                status='active'
            ).select_related('fruit_type')
            
            if not batches.exists():
                return {'message': 'No active fruit batches in this location'}
            
            fruits = [batch.fruit_type.name for batch in batches]
            
            # Check compatibility
            compatibility_issues = []
            for i, fruit1 in enumerate(fruits):
                for fruit2 in fruits[i+1:]:
                    if not self.ethylene_monitor.check_compatibility(fruit1, fruit2):
                        compatibility_issues.append({
                            'fruit1': fruit1,
                            'fruit2': fruit2,
                            'issue': 'Ethylene incompatibility'
                        })
            
            # Get storage conditions
            sensor_data = RealTimeSensorData.objects.filter(
                location=location
            ).order_by('-recorded_at')[:10]
            
            avg_conditions = {}
            if sensor_data.exists():
                temps = [s.value for s in sensor_data if s.sensor_type == 'temperature']
                humids = [s.value for s in sensor_data if s.sensor_type == 'humidity']
                
                avg_conditions = {
                    'temperature': np.mean(temps) if temps else None,
                    'humidity': np.mean(humids) if humids else None
                }
            
            return {
                'storage_location': location.name,
                'fruit_batches': [b.batch_number for b in batches],
                'compatibility_issues': compatibility_issues,
                'current_conditions': avg_conditions,
                'recommendations': self._generate_storage_recommendations(
                    batches, avg_conditions
                )
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_storage_recommendations(self, batches, conditions):
        """Generate storage recommendations for multiple batches"""
        recommendations = []
        
        if not conditions.get('temperature') or not conditions.get('humidity'):
            return ["Install sensors to monitor storage conditions"]
        
        temp = conditions['temperature']
        humidity = conditions['humidity']
        
        for batch in batches:
            fruit = batch.fruit_type
            
            if temp < fruit.optimal_temp_min:
                recommendations.append(
                    f"Increase temperature for {fruit.name} to optimal range"
                )
            elif temp > fruit.optimal_temp_max:
                recommendations.append(
                    f"Decrease temperature for {fruit.name} to optimal range"
                )
            
            if humidity < fruit.optimal_humidity_min:
                recommendations.append(
                    f"Increase humidity for {fruit.name} to optimal range"
                )
            elif humidity > fruit.optimal_humidity_max:
                recommendations.append(
                    f"Decrease humidity for {fruit.name} to optimal range"
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _find_latest_fruit_model(self):
        """Find the latest trained fruit quality model"""
        model_dir = os.path.join(settings.MEDIA_ROOT, 'fruit_models')
        if not os.path.exists(model_dir):
            return None
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            return None
        
        # Get the latest model
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        return os.path.join(model_dir, latest_model)


class RealProductAIService:
    """AI service for product anomaly detection and sensor analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        if SKLEARN_AVAILABLE:
            self.load_trained_models()
        else:
            print("AI Service running in simplified mode (scikit-learn not available)")
    
    def load_trained_models(self):
        """Load pre-trained models from database"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            from bika.models import TrainedModel
            active_models = TrainedModel.objects.filter(is_active=True)
            for model_obj in active_models:
                model_path = os.path.join(settings.MEDIA_ROOT, str(model_obj.model_file))
                if os.path.exists(model_path):
                    self.models[model_obj.model_type] = joblib.load(model_path)
                    print(f"Loaded model: {model_obj.model_type}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def train_anomaly_detection_model(self, dataset_id):
        """Train anomaly detection model on real dataset"""
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available. Cannot train models.")
            return None
            
        try:
            from bika.models import ProductDataset, TrainedModel
            dataset = ProductDataset.objects.get(id=dataset_id, dataset_type='anomaly_detection')
            dataset_path = os.path.join(settings.MEDIA_ROOT, str(dataset.data_file))
            
            # Load real dataset
            df = pd.read_csv(dataset_path)
            
            # Prepare features (adjust based on your dataset columns)
            feature_columns = ['stock_quantity', 'sales_velocity', 'return_rate', 
                             'defect_rate', 'shelf_life_days']
            
            # Use only existing columns
            available_features = [col for col in feature_columns if col in df.columns]
            X = df[available_features]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
            model.fit(X_scaled)
            
            # Save model
            model_filename = f"anomaly_model_{dataset_id}_{datetime.now().strftime('%Y%m%d')}.pkl"
            model_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', model_filename)
            joblib.dump(model, model_path)
            
            # Save to database
            trained_model = TrainedModel.objects.create(
                name=f"Anomaly Detection Model - {dataset.name}",
                model_type='anomaly_detection',
                dataset=dataset,
                model_file=f'trained_models/{model_filename}',
                feature_columns=available_features
            )
            
            self.models['anomaly_detection'] = model
            self.scalers['anomaly_detection'] = scaler
            
            return trained_model
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def detect_product_anomalies(self, product_data):
        """Detect anomalies in real product data"""
        if not SKLEARN_AVAILABLE:
            # Simplified anomaly detection without scikit-learn
            anomalies = []
            for product in product_data:
                # Simple rule-based anomaly detection
                if (hasattr(product, 'stock_quantity') and 
                    product.stock_quantity > 0 and 
                    product.stock_quantity <= getattr(product, 'low_stock_threshold', 5)):
                    anomalies.append({
                        'product_id': product.id,
                        'anomaly_score': 0.8,
                        'features': {'stock_quantity': product.stock_quantity},
                        'reason': 'Low stock detected'
                    })
            return anomalies
        
        if 'anomaly_detection' not in self.models:
            return []
        
        try:
            # Convert product data to features
            features = []
            product_ids = []
            
            for product in product_data:
                feature_vector = []
                for feature in self.scalers['anomaly_detection'].feature_names_in_:
                    feature_vector.append(getattr(product, feature, 0))
                features.append(feature_vector)
                product_ids.append(product.id)
            
            # Scale features
            features_scaled = self.scalers['anomaly_detection'].transform(features)
            
            # Predict anomalies
            predictions = self.models['anomaly_detection'].predict(features_scaled)
            anomaly_scores = self.models['anomaly_detection'].decision_function(features_scaled)
            
            # Return anomalies
            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1:  # Anomaly detected
                    anomalies.append({
                        'product_id': product_ids[i],
                        'anomaly_score': score,
                        'features': dict(zip(self.scalers['anomaly_detection'].feature_names_in_, features[i]))
                    })
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return []
    
    def analyze_sensor_data(self, sensor_readings):
        """Analyze real sensor data for quality issues"""
        alerts = []
        
        for reading in sensor_readings:
            # Define normal ranges based on product type
            normal_ranges = self.get_normal_ranges(reading.product)
            
            if reading.sensor_type in normal_ranges:
                min_val, max_val = normal_ranges[reading.sensor_type]
                
                if reading.value < min_val or reading.value > max_val:
                    alert_type = self.determine_alert_type(reading.sensor_type, reading.value, min_val, max_val)
                    severity = self.determine_severity(reading.sensor_type, reading.value, min_val, max_val)
                    
                    alerts.append({
                        'product': reading.product,
                        'sensor_type': reading.sensor_type,
                        'value': reading.value,
                        'normal_range': f"{min_val}-{max_val}",
                        'alert_type': alert_type,
                        'severity': severity,
                        'message': self.generate_alert_message(reading, alert_type, severity)
                    })
        
        return alerts
    
    def get_normal_ranges(self, product):
        """Get normal sensor ranges for specific product type"""
        # Define based on your product categories
        ranges = {
            'temperature': (15, 25),  # Celsius
            'humidity': (30, 70),     # Percentage
            'weight': (0.95, 1.05),   # Ratio to expected
            'vibration': (0, 5),      # Intensity
            'pressure': (95, 105),    # kPa
        }
        
        # Adjust ranges based on product category
        if hasattr(product, 'category') and product.category:
            category_name = product.category.name.lower()
            if 'food' in category_name:
                ranges['temperature'] = (0, 5)  # Refrigerated
            elif 'electronic' in category_name:
                ranges['humidity'] = (20, 50)   # Lower humidity
            elif 'fragile' in category_name:
                ranges['vibration'] = (0, 2)    # Lower vibration tolerance
        
        return ranges
    
    def determine_alert_type(self, sensor_type, value, min_val, max_val):
        """Determine the type of alert based on sensor reading"""
        alert_types = {
            'temperature': 'temperature_anomaly',
            'humidity': 'humidity_issue',
            'weight': 'weight_discrepancy',
            'vibration': 'vibration_alert',
            'pressure': 'pressure_anomaly'
        }
        return alert_types.get(sensor_type, 'sensor_anomaly')
    
    def determine_severity(self, sensor_type, value, min_val, max_val):
        """Determine severity based on deviation from normal range"""
        range_width = max_val - min_val
        deviation = min(abs(value - min_val), abs(value - max_val))
        
        if deviation > range_width * 0.5:
            return 'critical'
        elif deviation > range_width * 0.3:
            return 'high'
        elif deviation > range_width * 0.1:
            return 'medium'
        else:
            return 'low'
    
    def generate_alert_message(self, reading, alert_type, severity):
        """Generate descriptive alert message"""
        messages = {
            'temperature_anomaly': f"Temperature anomaly detected: {reading.value}°C",
            'humidity_issue': f"Humidity issue: {reading.value}%",
            'weight_discrepancy': f"Weight discrepancy: {reading.value}",
            'vibration_alert': f"Unusual vibration detected: {reading.value}",
            'pressure_anomaly': f"Pressure anomaly: {reading.value}"
        }
        
        base_message = messages.get(alert_type, f"Sensor anomaly: {reading.sensor_type} = {reading.value}")
        return f"{severity.upper()} - {base_message}"


# ==================== CREATE SERVICE INSTANCES ====================

# Create global instances for easy access
fruit_ai_service = FruitAIService()
product_ai_service = RealProductAIService()

# ==================== EXPORTS ====================

__all__ = [
    'FruitQualityPredictor',
    'FruitRipenessPredictor',
    'EthyleneMonitor',
    'FruitAIService',
    'RealProductAIService',
    'fruit_ai_service',
    'product_ai_service'
]