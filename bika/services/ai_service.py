# bika/services/ai_service.py - UPDATED FOR FIXED_REAL_WORLD_TRAINER
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

# Import the FixedRealWorldTrainer
from bika.fixed_real_world_trainer import FixedRealWorldTrainer

# Set up logger
logger = logging.getLogger(__name__)

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

# ==================== ENHANCED AI SERVICE ====================

class EnhancedFruitAIService:
    """Enhanced AI service using FixedRealWorldTrainer"""
    
    def __init__(self):
        self.trainer = FixedRealWorldTrainer()
        self.active_model = None
        self.load_active_model()
        self.prediction_history = []
        print("Enhanced Fruit AI Service initialized (using FixedRealWorldTrainer)")
    
    def load_active_model(self):
        """Load the active trained model"""
        try:
            from bika.models import TrainedModel
            active_model = TrainedModel.objects.filter(
                is_active=True, 
                model_type='quality'
            ).first()
            
            if active_model and os.path.exists(active_model.model_file.path):
                model_data = joblib.load(active_model.model_file.path)
                self.active_model = {
                    'model': model_data['model'],
                    'scaler': model_data.get('scaler'),
                    'label_encoder': model_data.get('label_encoder'),
                    'metadata': model_data.get('metadata', {}),
                    'record': active_model
                }
                print(f"✅ Loaded active model: {active_model.name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def predict_and_alert(self, product_id, sensor_data=None):
        """
        Make prediction and generate alerts for a product
        """
        try:
            from bika.models import Product, ProductAlert
            
            product = Product.objects.get(id=product_id)
            
            # Get sensor data if not provided
            if sensor_data is None:
                sensor_data = self._get_product_sensor_data(product)
            
            # Make prediction
            prediction = self.predict_quality(
                product.fruit_type if hasattr(product, 'fruit_type') else 'Apple',
                sensor_data['temperature'],
                sensor_data['humidity'],
                sensor_data.get('light_intensity', 50),
                sensor_data.get('co2_level', 400)
            )
            
            # Generate alerts based on prediction
            alerts = self._generate_alerts(prediction, product, sensor_data)
            
            # Save alerts to database
            for alert_data in alerts:
                ProductAlert.objects.create(
                    product=product,
                    alert_type=alert_data['type'],
                    severity=alert_data['priority'],
                    message=alert_data['message'],
                    details=json.dumps({
                        'predicted_quality': prediction.get('predicted_class'),
                        'confidence': prediction.get('confidence'),
                        'sensor_data': sensor_data,
                        'recommendations': alert_data.get('recommendations', [])
                    })
                )
            
            return {
                'success': True,
                'prediction': prediction,
                'alerts': alerts,
                'product': product.name
            }
            
        except Exception as e:
            logger.error(f"Error in predict_and_alert: {e}")
            return {'error': str(e)}
    
    def _get_product_sensor_data(self, product):
        """Get sensor data for a product"""
        try:
            from bika.models import FruitBatch, FruitQualityReading
            batch = FruitBatch.objects.filter(product=product, status='active').first()
            if batch:
                reading = FruitQualityReading.objects.filter(fruit_batch=batch).order_by('-timestamp').first()
                if reading:
                    return {
                        'temperature': float(reading.temperature),
                        'humidity': float(reading.humidity),
                        'light_intensity': float(reading.light_intensity),
                        'co2_level': float(reading.co2_level)
                    }
        except:
            pass
        
        # Return default values
        return {
            'temperature': 5.0,
            'humidity': 90.0,
            'light_intensity': 50.0,
            'co2_level': 400.0
        }
    
    def _generate_alerts(self, prediction, product, sensor_data):
        """Generate alerts based on prediction"""
        alerts = []
        
        quality = prediction.get('predicted_class', '').lower()
        confidence = prediction.get('confidence', 0)
        
        # Quality-based alerts
        if quality in ['bad', 'poor', 'rotten']:
            alerts.append({
                'type': 'quality_issue',
                'priority': 'critical' if quality == 'rotten' else 'high',
                'title': f'{product.name} Quality Alert',
                'message': f'Quality predicted as {quality.title()} with {confidence:.0%} confidence.',
                'recommendations': [
                    'Move to discount section',
                    'Check for contamination',
                    'Consider immediate processing'
                ]
            })
        
        # Temperature alerts
        if sensor_data['temperature'] > 12:
            alerts.append({
                'type': 'temperature_anomaly',
                'priority': 'high',
                'title': f'High Temperature Alert',
                'message': f'Temperature ({sensor_data["temperature"]}°C) is above optimal range.',
                'recommendations': [
                    'Adjust cooling system',
                    'Move to cooler location',
                    'Monitor closely'
                ]
            })
        
        # Humidity alerts
        if sensor_data['humidity'] < 85 or sensor_data['humidity'] > 95:
            alerts.append({
                'type': 'humidity_issue',
                'priority': 'medium',
                'title': f'Humidity Alert',
                'message': f'Humidity ({sensor_data["humidity"]}%) is outside optimal range (85-95%).',
                'recommendations': [
                    'Adjust humidity controls',
                    'Check ventilation'
                ]
            })
        
        return alerts
    
    def predict_quality(self, fruit_name, temperature, humidity, light_intensity, co2_level):
        """Predict fruit quality"""
        # Use active model if available
        if self.active_model:
            try:
                model = self.active_model['model']
                scaler = self.active_model['scaler']
                le = self.active_model['label_encoder']
                metadata = self.active_model['metadata']
                
                # Prepare features based on model's feature columns
                feature_names = metadata.get('feature_columns', [])
                
                if not feature_names:
                    # Default feature preparation
                    fruit_code = self._fruit_to_code(fruit_name)
                    features = [fruit_code, temperature, humidity, light_intensity, co2_level]
                else:
                    # Use model's specific feature columns
                    features = self._prepare_features_for_model(fruit_name, temperature, humidity, 
                                                               light_intensity, co2_level, feature_names)
                
                # Scale features
                if scaler:
                    features_scaled = scaler.transform([features])
                else:
                    features_scaled = [features]
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
                
                # Decode if label encoder exists
                if le:
                    try:
                        predicted_class = le.inverse_transform([prediction])[0]
                    except:
                        predicted_class = str(prediction)
                else:
                    predicted_class = str(prediction)
                
                # Get confidence
                confidence = prediction_proba.max() if prediction_proba is not None else 0.8
                
                # Calculate quality score
                quality_scores = {'Fresh': 100, 'Good': 80, 'Fair': 60, 'Poor': 30, 'Rotten': 0}
                quality_score = quality_scores.get(predicted_class, 50)
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': float(confidence),
                    'quality_score': quality_score,
                    'model_used': self.active_model['record'].name if self.active_model.get('record') else 'Active Model'
                }
                
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                # Fallback to rule-based prediction
        
        # Rule-based prediction (fallback)
        return self._rule_based_prediction(fruit_name, temperature, humidity, light_intensity, co2_level)
    
    def _fruit_to_code(self, fruit_name):
        """Convert fruit name to numeric code"""
        fruit_map = {
            'banana': 0, 'apple': 1, 'orange': 2, 'mango': 3,
            'tomato': 4, 'strawberry': 5, 'pineapple': 6,
            'grapes': 7, 'watermelon': 8, 'avocado': 9
        }
        return fruit_map.get(fruit_name.lower(), 0)
    
    def _prepare_features_for_model(self, fruit_name, temp, humidity, light, co2, feature_names):
        """Prepare features based on model's expected feature columns"""
        features = []
        
        for feature in feature_names:
            if feature.lower() == 'fruit_type' or feature.lower() == 'fruit':
                features.append(self._fruit_to_code(fruit_name))
            elif feature.lower() == 'temperature':
                features.append(temp)
            elif feature.lower() == 'humidity':
                features.append(humidity)
            elif feature.lower() == 'light_intensity' or feature.lower() == 'light':
                features.append(light)
            elif feature.lower() == 'co2_level' or feature.lower() == 'co2':
                features.append(co2)
            else:
                # Unknown feature, use default value
                features.append(0.0)
        
        return features
    
    def _rule_based_prediction(self, fruit_name, temperature, humidity, light, co2):
        """Rule-based quality prediction (fallback)"""
        quality_score = 80
        
        # Temperature effect
        if 2 <= temperature <= 8:
            quality_score += 10
        elif temperature < 0 or temperature > 12:
            quality_score -= 20
        else:
            quality_score -= 5
        
        # Humidity effect
        if 85 <= humidity <= 95:
            quality_score += 10
        elif humidity < 70 or humidity > 98:
            quality_score -= 15
        else:
            quality_score -= 5
        
        # CO2 effect
        if co2 > 1000:
            quality_score -= 10
        
        # Light effect
        if light > 100:
            quality_score -= 5
        
        # Determine quality class
        if quality_score >= 90:
            predicted_class = 'Fresh'
            confidence = 0.85
        elif quality_score >= 80:
            predicted_class = 'Good'
            confidence = 0.75
        elif quality_score >= 65:
            predicted_class = 'Fair'
            confidence = 0.65
        elif quality_score >= 40:
            predicted_class = 'Poor'
            confidence = 0.55
        else:
            predicted_class = 'Rotten'
            confidence = 0.45
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'quality_score': quality_score,
            'model_used': 'Rule-based'
        }
    
    def train_five_models(self, csv_file, target_column='quality_class'):
        """Train 5 different models using FixedRealWorldTrainer and select best one"""
        try:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                for chunk in csv_file.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Use FixedRealWorldTrainer
            df = self.trainer.load_dataset(tmp_path)
            
            if df is None:
                os.unlink(tmp_path)
                return {'error': 'Failed to load dataset'}
            
            # Analyze dataset
            analysis = self.trainer.analyze_dataset(df)
            
            # Check if target column exists
            if target_column not in df.columns:
                os.unlink(tmp_path)
                return {'error': f'Target column "{target_column}" not found in dataset'}
            
            # Preprocess for target
            X, y = self.trainer.preprocess_for_target(df, target_column)
            
            # Train models (this automatically trains 5 models)
            model_metrics = self.trainer.train_models(X, y)
            
            # Display results
            self.trainer.display_results()
            
            # Save best model to database
            if self.trainer.best_model:
                from bika.models import TrainedModel
                from django.core.files import File
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_filename = f'fruit_model_{timestamp}.pkl'
                
                # Prepare model data
                model_data = {
                    'model': self.trainer.best_model['model'],
                    'scaler': self.trainer.scaler,
                    'label_encoder': self.trainer.label_encoder,
                    'metadata': {
                        'target_column': target_column,
                        'feature_columns': self.trainer.feature_columns,
                        'training_date': timestamp,
                        'best_model_name': self.trainer.best_model['name'],
                        'best_model_key': self.trainer.best_model['key'],
                        'model_metrics': self.trainer.best_model['metrics'],
                        'problem_type': self.trainer.problem_type,
                        'dataset_info': analysis,
                        'all_models': model_metrics
                    }
                }
                
                # Save model file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_model:
                    joblib.dump(model_data, tmp_model.name)
                    
                    # Create TrainedModel record
                    trained_model = TrainedModel.objects.create(
                        name=f'Fruit Quality Model v{timestamp}',
                        model_type='quality',
                        version='1.0',
                        accuracy=self.trainer.best_model['metrics'].get('accuracy', 0) 
                            if self.trainer.problem_type == 'classification' 
                            else self.trainer.best_model['metrics'].get('r2_score', 0),
                        features_used=self.trainer.feature_columns
                    )
                    
                    # Save model file
                    with open(tmp_model.name, 'rb') as f:
                        trained_model.model_file.save(model_filename, File(f))
                    
                    # Deactivate other models
                    TrainedModel.objects.filter(
                        model_type='quality'
                    ).exclude(id=trained_model.id).update(is_active=False)
                
                # Clean up temp files
                os.unlink(tmp_model.name)
                os.unlink(tmp_path)
                
                # Reload active model
                self.load_active_model()
                
                # Prepare comprehensive result
                result = {
                    'success': True,
                    'model_saved': True,
                    'model_id': trained_model.id,
                    'best_model': {
                        'name': self.trainer.best_model['name'],
                        'key': self.trainer.best_model['key'],
                        'metrics': self.trainer.best_model['metrics']
                    },
                    'problem_type': self.trainer.problem_type,
                    'all_models': model_metrics,
                    'dataset_info': analysis,
                    'feature_columns': self.trainer.feature_columns,
                    'training_samples': X.shape[0]
                }
                
                # Add performance metrics
                if self.trainer.problem_type == 'classification':
                    result['accuracy'] = self.trainer.best_model['metrics']['accuracy']
                    result['f1_score'] = self.trainer.best_model['metrics']['f1_score']
                else:
                    result['r2_score'] = self.trainer.best_model['metrics']['r2_score']
                    result['rmse'] = self.trainer.best_model['metrics']['rmse']
                
                # Also save results locally
                self.trainer.save_results(df)
                
                return result
            else:
                os.unlink(tmp_path)
                return {'error': 'No model was trained successfully'}
            
        except Exception as e:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.error(f"Error training models: {e}")
            return {'error': str(e)}
    
    def get_detailed_model_comparison(self):
        """Get detailed comparison of all trained models"""
        try:
            from bika.models import TrainedModel
            models = TrainedModel.objects.filter(model_type='quality').order_by('-id')
            
            comparison_data = []
            for model in models:
                if os.path.exists(model.model_file.path):
                    try:
                        model_data = joblib.load(model.model_file.path)
                        metadata = model_data.get('metadata', {})
                        
                        comparison_data.append({
                            'id': model.id,
                            'name': model.name,
                            'version': model.version,
                            'accuracy': model.accuracy,
                            'created_at': model.created_at if hasattr(model, 'created_at') else None,
                            'is_active': model.is_active,
                            'features_count': len(model.features_used) if model.features_used else 0,
                            'metadata': {
                                'best_model': metadata.get('best_model_name', 'Unknown'),
                                'problem_type': metadata.get('problem_type', 'Unknown'),
                                'target_column': metadata.get('target_column', 'Unknown'),
                                'training_date': metadata.get('training_date', 'Unknown')
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error loading model {model.id}: {e}")
                        continue
            
            return {
                'success': True,
                'models': comparison_data,
                'total_models': len(comparison_data),
                'active_model': next((m for m in comparison_data if m['is_active']), None)
            }
            
        except Exception as e:
            logger.error(f"Error getting model comparison: {e}")
            return {'error': str(e)}
    
    def generate_sample_dataset(self, num_samples=1000):
        """Generate sample fruit quality dataset"""
        try:
            import numpy as np
            from datetime import datetime, timedelta
            
            np.random.seed(42)
            
            # Create realistic fruit quality dataset
            data = []
            for i in range(num_samples):
                # Random fruit type
                fruit_type = np.random.choice(['Banana', 'Apple', 'Orange', 'Mango', 'Tomato', 'Strawberry', 'Grapes'])
                
                # Storage conditions
                temperature = np.random.normal(5, 3)  # Mean 5°C, std 3
                humidity = np.random.normal(90, 5)    # Mean 90%, std 5
                light_intensity = np.random.uniform(0, 200)
                co2_level = np.random.uniform(300, 1500)
                
                # Determine quality based on conditions
                quality_score = 100
                
                # Temperature effect
                if 2 <= temperature <= 8:
                    quality_score += 10
                elif temperature < 0 or temperature > 12:
                    quality_score -= 20
                
                # Humidity effect
                if 85 <= humidity <= 95:
                    quality_score += 10
                elif humidity < 75 or humidity > 98:
                    quality_score -= 15
                
                # CO2 effect
                if co2_level > 1000:
                    quality_score -= 10
                
                # Light effect
                if light_intensity > 100:
                    quality_score -= 5
                
                # Add random variation
                quality_score += np.random.normal(0, 5)
                quality_score = max(0, min(100, quality_score))
                
                # Determine quality class
                if quality_score >= 90:
                    quality_class = 'Fresh'
                elif quality_score >= 80:
                    quality_class = 'Good'
                elif quality_score >= 65:
                    quality_class = 'Fair'
                elif quality_score >= 40:
                    quality_class = 'Poor'
                else:
                    quality_class = 'Rotten'
                
                # Add to dataset
                data.append({
                    'fruit_type': fruit_type,
                    'temperature': round(temperature, 2),
                    'humidity': round(humidity, 2),
                    'light_intensity': round(light_intensity, 2),
                    'co2_level': round(co2_level, 2),
                    'quality_class': quality_class,
                    'quality_score': round(quality_score, 2),
                    'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(data)
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"fruit_sample_dataset_{timestamp}.csv"
            
            # Create datasets directory if it doesn't exist
            dataset_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
            os.makedirs(dataset_dir, exist_ok=True)
            
            filepath = os.path.join(dataset_dir, filename)
            df.to_csv(filepath, index=False)
            
            return {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'samples': len(df),
                'columns': list(df.columns),
                'download_url': f"/media/datasets/{filename}",
                'preview': df.head(10).to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error generating sample dataset: {e}")
            return {'error': str(e)}
    
    def validate_dataset(self, csv_file):
        """Validate dataset before training"""
        try:
            # Save file temporarily
            import tempfile
            timestamp = int(timezone.now().timestamp())
            temp_path = os.path.join('temp_datasets', f'validate_{timestamp}.csv')
            saved_path = default_storage.save(temp_path, csv_file)
            full_path = default_storage.path(saved_path)
            
            # Load dataset using FixedRealWorldTrainer
            df = self.trainer.load_dataset(full_path)
            
            if df is None:
                return {'error': 'Failed to load dataset'}
            
            # Analyze dataset
            analysis = self.trainer.analyze_dataset(df)
            
            validation_results = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'columns': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'analysis': analysis
            }
            
            # Check for required columns for fruit quality prediction
            required_cols = ['temperature', 'humidity', 'light_intensity', 'co2_level', 'fruit_type', 'quality_class']
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                validation_results['missing_required_columns'] = missing_required
                validation_results['valid_for_training'] = False
            else:
                validation_results['missing_required_columns'] = []
                validation_results['valid_for_training'] = True
                
                # Additional quality checks
                validation_results['quality_class_distribution'] = df['quality_class'].value_counts().to_dict()
                validation_results['fruit_type_distribution'] = df['fruit_type'].value_counts().to_dict()
                
                # Numeric value ranges
                numeric_cols = ['temperature', 'humidity', 'light_intensity', 'co2_level']
                for col in numeric_cols:
                    if col in df.columns:
                        validation_results[f'{col}_range'] = {
                            'min': float(df[col].min()),
                            'max': float(df[col].max()),
                            'mean': float(df[col].mean()),
                            'std': float(df[col].std())
                        }
            
            # Calculate data quality score
            quality_score = 100
            
            # Penalize missing values
            missing_pct = validation_results['missing_values'] / (validation_results['total_rows'] * validation_results['total_columns']) * 100
            quality_score -= missing_pct * 2
            
            # Penalize duplicates
            duplicate_pct = validation_results['duplicate_rows'] / validation_results['total_rows'] * 100
            quality_score -= duplicate_pct
            
            # Penalize insufficient data
            if validation_results['total_rows'] < 50:
                quality_score -= 30
            elif validation_results['total_rows'] < 100:
                quality_score -= 15
            
            validation_results['data_quality_score'] = max(0, min(100, quality_score))
            
            # Recommendations
            recommendations = []
            if validation_results['missing_values'] > 0:
                recommendations.append(f"Remove or impute {validation_results['missing_values']} missing values")
            
            if validation_results['duplicate_rows'] > 0:
                recommendations.append(f"Remove {validation_results['duplicate_rows']} duplicate rows")
            
            if validation_results['total_rows'] < 100:
                recommendations.append(f"Dataset is small ({validation_results['total_rows']} rows). Consider collecting more data.")
            
            if missing_required:
                recommendations.append(f"Add missing columns: {', '.join(missing_required)}")
            
            validation_results['recommendations'] = recommendations
            
            # Clean up temporary file
            try:
                os.remove(full_path)
            except:
                pass
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return {'error': str(e)}
    
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
                'recommendations': self._generate_batch_recommendations(df, batch)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing batch trends: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, series):
        """Calculate trend (increasing, decreasing, stable)"""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
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
        quality_scores = df['quality_score'].values if 'quality_score' in df.columns else []
        if len(quality_scores) >= 2:
            if quality_scores[-1] < quality_scores[0] - 1:
                recommendations.append("Quality deterioration detected - consider priority sale")
        
        # Days remaining
        if hasattr(batch, 'days_remaining'):
            if batch.days_remaining <= 2:
                recommendations.append("URGENT: Batch approaching expiry - immediate action required")
            elif batch.days_remaining <= 5:
                recommendations.append("Batch nearing expiry - plan for sale or processing")
        
        return recommendations

# ==================== LEGACY AI MODELS (FOR COMPATIBILITY) ====================

class FruitQualityPredictor:
    """Legacy AI model for predicting fruit quality"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.class_names = ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
        
    def predict_quality(self, fruit_type, temperature, humidity, light_intensity, co2_level):
        """Legacy method - now uses EnhancedFruitAIService"""
        # Create instance of enhanced service
        service = EnhancedFruitAIService()
        return service.predict_quality(fruit_type, temperature, humidity, light_intensity, co2_level)

class FruitRipenessPredictor:
    """Predict fruit ripeness based on multiple factors"""
    
    def __init__(self):
        self.ripening_models = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default ripening models for common fruits"""
        self.ripening_rates = {
            'Banana': {'base_rate': 0.8, 'ethylene_factor': 1.5, 'temp_factor': 0.1},
            'Apple': {'base_rate': 0.3, 'ethylene_factor': 1.2, 'temp_factor': 0.05},
            'Orange': {'base_rate': 0.4, 'ethylene_factor': 1.1, 'temp_factor': 0.08},
            'Mango': {'base_rate': 1.0, 'ethylene_factor': 2.0, 'temp_factor': 0.15},
            'Tomato': {'base_rate': 0.9, 'ethylene_factor': 1.8, 'temp_factor': 0.12},
        }
    
    def predict_ripeness(self, fruit_type, temperature, ethylene_level, days_since_harvest):
        """Predict ripeness level"""
        fruit_type = fruit_type.capitalize()
        params = self.ripening_rates.get(fruit_type, {'base_rate': 0.5, 'ethylene_factor': 1.2, 'temp_factor': 0.08})
        
        # Calculate ripening score
        base_rate = params['base_rate']
        ethylene_factor = 1 + (params['ethylene_factor'] * ethylene_level / 100)
        temp_effect = 1 + (params['temp_factor'] * abs(temperature - 20) / 10)
        days_effect = 1 + (days_since_harvest / 10)
        
        ripeness_score = min(1.0, base_rate * ethylene_factor * temp_effect * days_effect)
        
        # Determine ripeness stage
        if ripeness_score < 0.3:
            stage = 'unripe'
        elif ripeness_score < 0.6:
            stage = 'ripe'
        elif ripeness_score < 0.8:
            stage = 'fully_ripe'
        else:
            stage = 'overripe'
        
        return {
            'ripeness_stage': stage,
            'ripeness_score': ripeness_score,
            'estimated_days_to_overripe': max(0, int((1.0 - ripeness_score) * 3))
        }

class EthyleneMonitor:
    """Monitor ethylene production and effects on fruits"""
    
    def __init__(self):
        self.ethylene_producers = {
            'Apple': 'high',
            'Banana': 'high',
            'Tomato': 'high',
            'Avocado': 'high',
            'Pear': 'medium',
        }
        
        self.ethylene_sensitive = {
            'Lettuce': 'very_high',
            'Broccoli': 'very_high',
            'Carrot': 'high',
            'Cucumber': 'high',
            'Watermelon': 'medium',
        }
    
    def check_compatibility(self, fruit1, fruit2):
        """Check if two fruits can be stored together"""
        fruit1 = fruit1.capitalize()
        fruit2 = fruit2.capitalize()
        
        if fruit1 == fruit2:
            return True, "Same fruit type - compatible"
        
        producer1 = self.ethylene_producers.get(fruit1)
        producer2 = self.ethylene_producers.get(fruit2)
        sensitive1 = self.ethylene_sensitive.get(fruit1)
        sensitive2 = self.ethylene_sensitive.get(fruit2)
        
        # Check for incompatibility
        if producer1 and sensitive2:
            return False, f"{fruit1} (ethylene producer) incompatible with {fruit2} (ethylene sensitive)"
        
        if producer2 and sensitive1:
            return False, f"{fruit2} (ethylene producer) incompatible with {fruit1} (ethylene sensitive)"
        
        return True, "Compatible for storage"

# ==================== GLOBAL INSTANCE ====================

# Create global AI service instance
enhanced_ai_service = EnhancedFruitAIService()

# Export for compatibility
__all__ = [
    'EnhancedFruitAIService',
    'FruitQualityPredictor',
    'FruitRipenessPredictor',
    'EthyleneMonitor',
    'enhanced_ai_service'
]