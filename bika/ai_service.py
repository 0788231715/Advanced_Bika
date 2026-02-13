# bika/services/ai_service.py - AI SERVICE LAYER FOR BIKA APPLICATION
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

import joblib # Added for model loading
from django.db import transaction # Added for atomic operations
from django.db.models import Avg # Added
from django.core.mail import send_mail # Added for sending alert notifications
from bika.models import (
    Product, FruitBatch, StorageLocation, CustomUser # Replaced FruitProduct, Batch, Warehouse, added CustomUser
)
from bika.ai_integration.models import (
    FruitPrediction, AlertNotification # For saving predictions and alerts
)
from bika.models import TrainedModel # For loading active trained models

# Import AI models from the main models module
from bika.ai_models import (
    FruitQualityPredictor, FruitRipenessPredictor,
    EthyleneMonitor, FruitDiseasePredictor, FruitPricePredictor,
    ShelfLifePredictor, # Added
    BikaAIService
)

# Set up logger
logger = logging.getLogger(__name__)

# ==================== ENHANCED AI SERVICE ====================

class EnhancedBikaAIService(BikaAIService):
    """Enhanced AI service with additional features and integrations"""
    
    def __init__(self):
        super().__init__()
        self.data_cache = {}
        self.model_cache = {}
        self.prediction_history = []
        self.loaded_predictors = {} # Initialize dictionary to hold loaded predictor instances
        self._load_all_active_ai_models() # Call new method to load all active AI models
        self.shelf_life_predictor = ShelfLifePredictor() # Added for fallback

        
    def _load_all_active_ai_models(self):
        """Loads all active trained models from the database into loaded_predictors."""
        self.loaded_predictors = {} # Clear existing
        active_models = TrainedModel.objects.filter(is_active=True)

        for model_record in active_models:
            try:
                model_path = model_record.model_file.path
                if os.path.exists(model_path):
                    # Load model data
                    model_data = joblib.load(model_path)

                    predictor_instance = None
                    # Instantiate the correct predictor class based on model_type
                    if model_record.model_type == 'fruit_quality': # This maps to FruitQualityPredictor in ai_models.py
                        predictor_instance = FruitQualityPredictor()
                    elif model_record.model_type == 'ripeness': # Assuming ripeness predictor exists
                        predictor_instance = FruitRipenessPredictor()
                    elif model_record.model_type == 'disease': # Assuming disease predictor exists
                        predictor_instance = FruitDiseasePredictor()
                    elif model_record.model_type == 'price': # Assuming price predictor exists
                                                    predictor_instance = FruitPricePredictor()
                                                elif model_record.model_type == 'shelf_life': # Added
                                                    predictor_instance = ShelfLifePredictor()
                                                elif model_record.model_type == 'ethylene': # Added
                                                    predictor_instance = EthyleneMonitor()                    # Add more model types as needed

                    if predictor_instance and hasattr(predictor_instance, 'load_model'):
                        # Load the specific model into the predictor instance
                        if predictor_instance.load_model(model_path):
                            self.loaded_predictors[model_record.model_type] = predictor_instance
                            logger.info(f"✅ Loaded {model_record.model_type} model: {model_record.name}")
                        else:
                            logger.warning(f"❌ Failed to load model data into predictor for {model_record.name}")
                    else:
                        logger.warning(f"❌ No suitable predictor class found or `load_model` method missing for model type: {model_record.model_type}")
                else:
                    logger.warning(f"❌ Model file not found for {model_record.name}: {model_path}")
            except Exception as e:
                logger.error(f"❌ Error loading model {model_record.name} ({model_record.model_type}): {e}")
        
        # Fallback for core predictors if not explicitly loaded as TrainedModels
        # This ensures basic functionality even if no TrainedModels are configured for them
        if 'fruit_quality' not in self.loaded_predictors:
            self.loaded_predictors['fruit_quality'] = self.quality_predictor # Retain existing quality_predictor instance
        if 'ripeness' not in self.loaded_predictors:
            self.loaded_predictors['ripeness'] = self.ripeness_predictor
        if 'disease' not in self.loaded_predictors:
            self.loaded_predictors['disease'] = self.disease_predictor
        if 'price' not in self.loaded_predictors:
            self.loaded_predictors['price'] = self.price_predictor
        if 'ethylene' not in self.loaded_predictors:
            self.loaded_predictors['ethylene'] = self.ethylene_monitor
        if 'shelf_life' not in self.loaded_predictors: # Added
            self.loaded_predictors['shelf_life'] = self.shelf_life_predictor # Added
        
        logger.info(f"Loaded predictors: {list(self.loaded_predictors.keys())}")


    def _get_latest_sensor_data(self, product: Product) -> Dict[str, float]:
        """
        Get latest sensor readings for a product.
        Prioritizes sensor data from the product's associated storage location,
        then falls back to product's own fields, or defaults.
        """
        # Try to get from product's storage location
        # Assuming Product has a ForeignKey to StorageLocation or can derive it
        # For InventoryItem, the location is directly linked.
        # For a general Product, we might need a more complex lookup,
        # but for now, we'll assume a direct link or a way to find relevant sensors.
        storage_location = getattr(product, 'storage_location', None) # Assuming a field like this or similar lookup
        
        if storage_location:
            # Import RealTimeSensorData dynamically to avoid circular dependencies if it's in bika.models
            from bika.models import RealTimeSensorData
            readings = RealTimeSensorData.objects.filter(
                location=storage_location,
                recorded_at__gte=timezone.now() - timedelta(hours=24)
            ).order_by('-recorded_at')
            
            # Aggregate sensor data for the location
            sensor_data = {
                'temperature': 22.0, 'humidity': 85.0, 'light_intensity': 50.0, 'co2_level': 400.0, 'ethylene_level': 0.0
            }
            if readings.exists():
                for reading in readings:
                    if reading.sensor_type == 'temperature':
                        sensor_data['temperature'] = float(reading.value)
                    elif reading.sensor_type == 'humidity':
                        sensor_data['humidity'] = float(reading.value)
                    elif reading.sensor_type == 'light': # Assuming 'light' for light_intensity
                        sensor_data['light_intensity'] = float(reading.value)
                    elif reading.sensor_type == 'co2':
                        sensor_data['co2_level'] = float(reading.value)
                    elif reading.sensor_type == 'ethylene':
                        sensor_data['ethylene_level'] = float(reading.value)
                return sensor_data
        
        # Fallback to product's own sensor data or defaults
        # Assuming Product model has fields like current_temperature, etc.
        # If not, these would need to be added or derived differently.
        return {
            'temperature': getattr(product, 'current_temperature', 22.0),
            'humidity': getattr(product, 'current_humidity', 85.0),
            'light_intensity': getattr(product, 'light_exposure', 50.0),
            'co2_level': getattr(product, 'co2_level', 400.0),
            'ethylene_level': getattr(product, 'ethylene_level', 0.0)
        }

    def _generate_quality_alerts(self, prediction_result: Dict[str, Any], product: Product, batch: Optional[FruitBatch] = None) -> List[Dict[str, Any]]:
        """Generate alerts based on quality prediction."""
        alerts = []
        
        if not prediction_result:
            return alerts
        
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        input_conditions = prediction_result['input_conditions'] # Contains temperature, humidity, etc.

        # Quality-based alerts
        quality_map = {
            'Rotten': {'priority': 'critical', 'message_suffix': 'This product requires immediate disposal or processing.'},
            'Poor': {'priority': 'high', 'message_suffix': 'Consider priority sale or processing at a discount.'},
            'Fair': {'priority': 'medium', 'message_suffix': 'Monitor closely and consider adjusted pricing.'},
        }

        if predicted_class in quality_map:
            info = quality_map[predicted_class]
            alerts.append({
                'type': 'quality_drop',
                'priority': info['priority'],
                'title': f'{product.name} Quality Alert: {predicted_class.upper()}',
                'message': f'Product "{product.name}" is predicted as {predicted_class} quality with {confidence:.0%} confidence. {info["message_suffix"]}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result, # Store full prediction data
            })

        # Environmental condition alerts (examples, can be expanded)
        if input_conditions.get('temperature') is not None and (input_conditions['temperature'] < 2 or input_conditions['temperature'] > 12):
            priority = 'critical' if input_conditions['temperature'] < 0 or input_conditions['temperature'] > 20 else 'high'
            alerts.append({
                'type': 'temperature_issue',
                'priority': priority,
                'title': f'High Temperature Anomaly for {product.name}',
                'message': f'Current temperature is {input_conditions["temperature"]}°C, which is outside optimal range. This could severely impact quality.',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        
        if input_conditions.get('humidity') is not None and (input_conditions['humidity'] < 70 or input_conditions['humidity'] > 95):
            priority = 'high' if input_conditions['humidity'] < 60 or input_conditions['humidity'] > 98 else 'medium'
            alerts.append({
                'type': 'humidity_issue',
                'priority': priority,
                'title': f'Humidity Anomaly for {product.name}',
                'message': f'Current humidity is {input_conditions["humidity"]}%, which is outside optimal range. Risk of dehydration or mold.',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })

        return alerts
    
    def _generate_disease_alerts(self, prediction_result: Dict[str, Any], product: Product, batch: Optional[FruitBatch] = None) -> List[Dict[str, Any]]:
        """Generate disease risk alerts based on disease prediction."""
        alerts = []
        if not prediction_result:
            return alerts

        risk_level = prediction_result.get('risk_level', 'Unknown').lower()
        
        if risk_level in ['high', 'critical']:
            alerts.append({
                'type': 'disease_risk',
                'priority': risk_level,
                'title': f'Disease Risk Alert for {product.name}: {risk_level.title()}',
                'message': f'Product "{product.name}" is at {risk_level} risk of disease. Recommendations: {", ".join(prediction_result.get("recommendations", []))}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        return alerts

    def _generate_shelf_life_alerts(self, prediction_result: Dict[str, Any], product: Product, batch: Optional[FruitBatch] = None) -> List[Dict[str, Any]]:
        """Generate alerts based on shelf life prediction."""
        alerts = []
        if not prediction_result:
            return alerts

        status = prediction_result.get('status', 'Optimal').lower()
        predicted_days = prediction_result.get('predicted_days_remaining', 999)
        recommendations = prediction_result.get('recommendations', [])

        if status == 'critical':
            alerts.append({
                'type': 'shelf_life_critical',
                'priority': 'critical',
                'title': f'CRITICAL Shelf Life Alert for {product.name} ({predicted_days} days left)',
                'message': f'Product "{product.name}" has only {predicted_days} days of shelf life remaining. Immediate action required. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        elif status == 'warning':
            alerts.append({
                'type': 'shelf_life_warning',
                'priority': 'high',
                'title': f'WARNING Shelf Life Alert for {product.name} ({predicted_days} days left)',
                'message': f'Product "{product.name}" has {predicted_days} days of shelf life remaining. Plan for quick turnover. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        
        # Also add environmental alerts if any from the shelf-life prediction's recommendations
        for rec in recommendations:
            if "Temperature is below optimal" in rec or "Temperature is above optimal" in rec:
                alerts.append({
                    'type': 'environmental_shelf_life_temp',
                    'priority': 'medium',
                    'title': f'Environmental Alert (Shelf Life) for {product.name}',
                    'message': rec,
                    'product': product,
                    'batch': batch,
                    'prediction_data': prediction_result,
                })
            elif "Humidity is low" in rec or "Humidity is high" in rec:
                alerts.append({
                    'type': 'environmental_shelf_life_hum',
                    'priority': 'medium',
                    'title': f'Environmental Alert (Shelf Life) for {product.name}',
                    'message': rec,
                    'product': product,
                    'batch': batch,
                    'prediction_data': prediction_result,
                })
            elif "Ethylene levels are elevated" in rec:
                alerts.append({
                    'type': 'environmental_shelf_life_ethylene',
                    'priority': 'high',
                    'title': f'Ethylene Alert (Shelf Life) for {product.name}',
                    'message': rec,
                    'product': product,
                    'batch': batch,
                    'prediction_data': prediction_result,
                })

        return alerts

    def _generate_ethylene_alerts(self, prediction_result: Dict[str, Any], product: Product, batch: Optional[FruitBatch] = None) -> List[Dict[str, Any]]:
        """Generate alerts based on ethylene risk prediction."""
        alerts = []
        if not prediction_result:
            return alerts

        risk_level = prediction_result.get('risk_level', 'Low').lower()
        concentration = prediction_result.get('ethylene_concentration_ppm', 0.0)
        recommendations = prediction_result.get('recommendations', [])

        if risk_level == 'critical':
            alerts.append({
                'type': 'ethylene_critical',
                'priority': 'critical',
                'title': f'CRITICAL Ethylene Alert for {product.name} ({concentration:.2f} ppm)',
                'message': f'Ethylene concentration for "{product.name}" is critical ({concentration:.2f} ppm). Immediate action required. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        elif risk_level == 'high':
            alerts.append({
                'type': 'ethylene_high_risk',
                'priority': 'high',
                'title': f'HIGH Risk Ethylene Alert for {product.name} ({concentration:.2f} ppm)',
                'message': f'Ethylene concentration for "{product.name}" is high ({concentration:.2f} ppm). Increase ventilation. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        elif risk_level == 'medium':
            alerts.append({
                'type': 'ethylene_medium_risk',
                'priority': 'medium',
                'title': f'MEDIUM Risk Ethylene Alert for {product.name} ({concentration:.2f} ppm)',
                'message': f'Ethylene concentration for "{product.name}" is elevated ({concentration:.2f} ppm). Monitor closely. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        
        # Add compatibility alerts if any
        compatibility_check_message = next((rec for rec in recommendations if "incompatible with" in rec), None)
        if compatibility_check_message:
             alerts.append({
                'type': 'ethylene_compatibility_issue',
                'priority': 'medium',
                'title': f'Ethylene Compatibility Warning for {product.name}',
                'message': compatibility_check_message,
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })

        return alerts

    def _generate_ripeness_alerts(self, prediction_result: Dict[str, Any], product: Product, batch: Optional[FruitBatch] = None) -> List[Dict[str, Any]]:
        """Generate alerts based on ripeness prediction."""
        alerts = []
        if not prediction_result:
            return alerts

        ripeness_stage = prediction_result.get('ripeness_stage', 'Unknown').lower()
        estimated_days_to_overripe = prediction_result.get('estimated_days_to_overripe', 999)
        recommendations = prediction_result.get('recommendations', [])

        if ripeness_stage == 'overripe':
            alerts.append({
                'type': 'ripeness_overripe_critical',
                'priority': 'critical',
                'title': f'CRITICAL Ripeness Alert for {product.name}: Overripe',
                'message': f'Product "{product.name}" is predicted to be overripe. Estimated days to overripe: {estimated_days_to_overripe}. Immediate processing or disposal recommended. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        elif estimated_days_to_overripe <= 2 and ripeness_stage != 'overripe':
            alerts.append({
                'type': 'ripeness_overripe_warning',
                'priority': 'high',
                'title': f'WARNING Ripeness Alert for {product.name}: Nearing Overripe',
                'message': f'Product "{product.name}" is {ripeness_stage}, but is nearing overripe stage ({estimated_days_to_overripe} days left). Prioritize for sale. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })
        elif ripeness_stage == 'unripe' and estimated_days_to_overripe < 5: # If unripe but spoilage is fast
             alerts.append({
                'type': 'ripeness_unripe_warning',
                'priority': 'medium',
                'title': f'Ripeness Alert for {product.name}: Unripe but Short Window',
                'message': f'Product "{product.name}" is unripe, but has limited time before overripening ({estimated_days_to_overripe} days). Monitor closely. Recommendations: {", ".join(recommendations)}',
                'product': product,
                'batch': batch,
                'prediction_data': prediction_result,
            })

        return alerts

    def _save_prediction(self, prediction_data: Dict[str, Any], product: Product, batch: Optional[FruitBatch] = None) -> Optional[FruitPrediction]:
        """Save prediction to database."""
        if not prediction_data:
            return None
        
        try:
            with transaction.atomic():
                prediction = FruitPrediction(
                    product=product,
                    batch=batch,
                    prediction_type=prediction_data['type'],
                    predicted_value=str(prediction_data['predicted_value']),
                    confidence=prediction_data.get('confidence', 0.0),
                    predicted_score=prediction_data.get('score'),
                    
                    # Sensor data
                    temperature=prediction_data.get('input_conditions', {}).get('temperature'),
                    humidity=prediction_data.get('input_conditions', {}).get('humidity'),
                    light_intensity=prediction_data.get('input_conditions', {}).get('light_intensity'),
                    co2_level=prediction_data.get('input_conditions', {}).get('co2_level'),
                    ethylene_level=prediction_data.get('input_conditions', {}).get('ethylene_level'),
                    
                    # Alert info (determine based on prediction)
                    alert_level=self._determine_alert_level(prediction_data),
                    alert_message=self._generate_alert_message(prediction_data),
                    recommendations=self._generate_recommendations_for_saving(prediction_data), # Use the unified recommendations method
                    
                    model_used=self._get_model_record_from_type(prediction_data['type']) # Link to the TrainedModel record
                )
                prediction.save()
                return prediction
                
        except Exception as e:
            logger.error(f"Error saving prediction for product {product.id}: {e}")
            return None

    def _determine_alert_level(self, prediction_data: Dict[str, Any]) -> str:
        """Determine alert level based on prediction data."""
        pred_type = prediction_data['type']
        pred_value = str(prediction_data['predicted_value']).lower()
        score = prediction_data.get('score', 0)
        confidence = prediction_data.get('confidence', 0)
        
        # Consider confidence in alert level determination
        if confidence < 0.5: # Low confidence might warrant a lower priority alert
            return 'info'

        if pred_type == 'quality':
            if pred_value in ['rotten', 'bad']:
                return 'critical'
            elif pred_value in ['poor']: # Changed 'fair' to 'poor' for higher priority
                return 'warning'
            elif pred_value == 'fair':
                return 'info' # Fair quality is just informational

        elif pred_type == 'disease':
            if score > 0.7:
                return 'critical'
            elif score > 0.4:
                return 'warning'

        elif pred_type == 'shelf_life':
            # Directly use score which is predicted_days_remaining
            days = prediction_data.get('predicted_days_remaining')
            if days is not None:
                if days <= 2:
                    return 'critical'
                elif days <= 5:
                    return 'warning'
            # Default to info if days are higher or not available
            return 'info'

        elif pred_type == 'ethylene':
            risk_level = prediction_data.get('risk_level', 'low').lower()
            if risk_level == 'critical':
                return 'critical'
            elif risk_level == 'high':
                return 'high'
            elif risk_level == 'medium':
                return 'warning'
            return 'info'

        elif pred_type == 'ripeness': # Added
            ripeness_stage = prediction_data.get('ripeness_stage', 'unknown').lower()
            estimated_days_to_overripe = prediction_data.get('estimated_days_to_overripe', 999)
            
            if ripeness_stage == 'overripe' or estimated_days_to_overripe <= 1:
                return 'critical'
            elif ripeness_stage == 'fully_ripe' or estimated_days_to_overripe <= 3:
                return 'warning'
            return 'info'
    def _generate_alert_message(self, prediction_data: Dict[str, Any]) -> str:
        """Generate a concise alert message from prediction data."""
        pred_type = prediction_data['type']
        pred_value = prediction_data['predicted_value']
        confidence = prediction_data.get('confidence', 0.0)
        product_name = prediction_data.get('product', Product()).name if prediction_data.get('product') else 'Unknown Product'

        messages = {
            'quality': f'Quality of {product_name} predicted as {pred_value} ({confidence:.0%} confidence).',
            'ripeness': f'Ripeness of {product_name} is {pred_value}.',
            'disease': f'High disease risk detected for {product_name}: {pred_value} ({confidence:.0%} confidence).',
            'shelf_life': f'{product_name} has {pred_value} remaining shelf life.',
            'price': f'Price recommendation for {product_name}: {pred_value}.',
            'ethylene': f'Ethylene risk for {product_name}: {prediction_data.get("risk_level", "Unknown")} ({prediction_data.get("ethylene_concentration_ppm", 0.0):.2f} ppm).'
        }
        
        message = messages.get(pred_type, f'AI Prediction for {product_name} ({pred_type}): {pred_value}.')
        if pred_type == 'shelf_life' and prediction_data.get('status') == 'Critical':
            message = f"CRITICAL: {message} Immediate action advised."
        elif pred_type == 'shelf_life' and prediction_data.get('status') == 'Warning':
            message = f"WARNING: {message} Plan for quick turnover."
        elif pred_type == 'ethylene' and prediction_data.get('risk_level') == 'Critical':
            message = f"CRITICAL: {message} Immediate ventilation required."
        elif pred_type == 'ethylene' and prediction_data.get('risk_level') == 'High':
            message = f"WARNING: {message} Increased ventilation recommended."
        elif pred_type == 'ripeness' and prediction_data.get('ripeness_stage') == 'overripe': # Added
            message = f"CRITICAL: {message} Product is overripe. Consider immediate processing or disposal."
        elif pred_type == 'ripeness' and prediction_data.get('ripeness_stage') == 'fully_ripe': # Added
            message = f"WARNING: {message} Product is fully ripe. Prioritize for sale."
        return message

    def _generate_recommendations_for_saving(self, prediction_data: Dict[str, Any]) -> List[str]:
        """
        Generates recommendations for saving to the FruitPrediction model.
        This consolidates recommendations from various predictors.
        """
        all_recommendations = []
        pred_type = prediction_data['type']
        
        # Prioritize recommendations from the AI models' own prediction results
        if prediction_data.get('recommendations'):
            all_recommendations.extend(prediction_data['recommendations'])
        
        # Add general recommendations based on prediction type and value
        pred_value = str(prediction_data['predicted_value']).lower()
        score = prediction_data.get('score', 0)

        if pred_type == 'quality':
            if pred_value in ['rotten', 'bad']:
                all_recommendations.extend(['Immediate disposal', 'Isolate from other products', 'Document loss'])
            elif pred_value == 'poor':
                all_recommendations.extend(['Offer discount', 'Prioritize for quick sale', 'Process into other products'])
            elif pred_value == 'fair':
                all_recommendations.extend(['Monitor closely', 'Adjust storage conditions if possible'])
        
        elif pred_type == 'disease':
            if score > 0.7:
                all_recommendations.extend(['Quarantine affected products', 'Sanitize storage area', 'Consider treatment options'])
        
        elif pred_type == 'shelf_life':
            # Use recommendations directly from the shelf_life_prediction_raw output
            if prediction_data.get('recommendations'):
                all_recommendations.extend(prediction_data['recommendations'])
        elif pred_type == 'ethylene':
            if prediction_data.get('recommendations'):
                all_recommendations.extend(prediction_data['recommendations'])
        elif pred_type == 'ripeness': # Added
            if prediction_data.get('recommendations'):
                all_recommendations.extend(prediction_data['recommendations'])
            ripeness_stage = prediction_data.get('ripeness_stage', 'unknown').lower()
            if ripeness_stage == 'overripe':
                all_recommendations.append('Immediately process or dispose of product.')
            elif ripeness_stage == 'fully_ripe':
                all_recommendations.append('Prioritize product for immediate sale.')
            elif ripeness_stage == 'unripe':
                all_recommendations.append('Monitor ripening process closely. Adjust storage conditions to accelerate/decelerate ripening if needed.')

        # Remove duplicates and return
        return list(set(all_recommendations))

    def _get_model_record_from_type(self, model_type: str) -> Optional[TrainedModel]:
        """Retrieve the active TrainedModel record for a given model type."""
        # Mapping from prediction_data type to TrainedModel.model_type
        type_mapping = {
            'quality': 'fruit_quality',
            'ripeness': 'ripeness', # Assuming this is a model_type in TrainedModel
            'disease': 'disease',   # Assuming this is a model_type in TrainedModel
            'price': 'price',       # Assuming this is a model_type in TrainedModel
            'shelf_life': 'shelf_life', # Added
            'ethylene': 'ethylene' # Added
        }
        trained_model_type = type_mapping.get(model_type)
        if trained_model_type:
            return TrainedModel.objects.filter(model_type=trained_model_type, is_active=True).first()
        return None

    def _save_alert(self, alert_data: Dict[str, Any], related_prediction: Optional[FruitPrediction] = None) -> Optional[AlertNotification]:
        """Save alert to database."""
        if not alert_data:
            return None
        
        try:
            with transaction.atomic():
                alert = AlertNotification(
                    alert_type=alert_data['type'],
                    priority=alert_data['priority'],
                    title=alert_data.get('title', 'AI Generated Alert'),
                    message=alert_data['message'],
                    product=alert_data.get('product'),
                    batch=alert_data.get('batch'),
                    # warehouse=alert_data.get('warehouse'), # Assuming warehouse might be passed
                    prediction=related_prediction
                )
                alert.save()
                return alert
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return None

    def _send_alert_notifications(self, alerts: List[AlertNotification]):
        """Send email notifications for critical alerts."""
        critical_alerts = [a for a in alerts if a.priority in ['critical', 'high']]
        
        if not critical_alerts:
            return
        
        try:
            admin_users = CustomUser.objects.filter(is_staff=True, is_active=True)
            admin_emails = [user.email for user in admin_users if user.email]
            
            if not admin_emails:
                logger.warning("No admin emails found to send critical alerts.")
                return
            
            subject = f"🚨 {len(critical_alerts)} Critical Fruit Quality Alerts"
            
            message_lines = ["CRITICAL ALERTS REQUIRING ATTENTION:\n"]
            
            for alert in critical_alerts:
                message_lines.append(f"🔴 {alert.title}")
                message_lines.append(f"   Product: {alert.product.name if alert.product else 'Unknown'}")
                message_lines.append(f"   Priority: {alert.priority.upper()}")
                message_lines.append(f"   Message: {alert.message}")
                message_lines.append(f"   Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                message_lines.append("")
            
            message_lines.append("\nPlease log in to the system to take action.")
            message = "\n".join(message_lines)
            
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=admin_emails,
                fail_silently=False # Set to False to log errors
            )
            
            logger.info(f"📧 Sent {len(critical_alerts)} critical alerts to {len(admin_emails)} admins")
            
        except Exception as e:
            logger.error(f"Error sending alert emails: {e}", exc_info=True)

    def predict_product_insights(self, product_id: int, batch_id: Optional[int] = None,
                                 sensor_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Orchestrates AI predictions for a given product, generating insights,
        alerts, and saving results to the database.
        """
        try:
            product = Product.objects.get(id=product_id)
            batch = FruitBatch.objects.get(id=batch_id) if batch_id else None
        except Product.DoesNotExist:
            logger.error(f"Product with ID {product_id} not found.")
            return {"error": f"Product with ID {product_id} not found."}
        except FruitBatch.DoesNotExist:
            logger.error(f"FruitBatch with ID {batch_id} not found.")
            return {"error": f"FruitBatch with ID {batch_id} not found."}
        
        # Calculate days_since_harvest for ripeness prediction
        days_since_harvest = 0
        if hasattr(product, 'harvest_date') and product.harvest_date:
            days_since_harvest = (timezone.now().date() - product.harvest_date).days
        elif hasattr(product, 'production_date') and product.production_date:
            days_since_harvest = (timezone.now().date() - product.production_date).days
        
        # Determine fruit_type for predictors
        fruit_type = product.fruit_type.name if hasattr(product.fruit_type, 'name') else str(product.name) # Use product name as fallback

        # 1. Fetch Sensor Data
        if sensor_data is None:
            sensor_data = self._get_latest_sensor_data(product)
        
        all_predictions_data = []
        all_generated_alerts = []
        
        input_conditions = {
            "temperature": sensor_data.get("temperature"),
            "humidity": sensor_data.get("humidity"),
            "light_intensity": sensor_data.get("light_intensity"),
            "co2_level": sensor_data.get("co2_level"),
            "ethylene_level": sensor_data.get("ethylene_level"),
        }

        # 2. Make Predictions using loaded_predictors
        # Quality Prediction
        if 'fruit_quality' in self.loaded_predictors:
            try:
                quality_predictor = self.loaded_predictors['fruit_quality']
                quality_prediction_raw = quality_predictor.predict_quality(
                    temperature=input_conditions["temperature"],
                    humidity=input_conditions["humidity"],
                    light_intensity=input_conditions["light_intensity"],
                    co2_level=input_conditions["co2_level"]
                )
                quality_prediction = {
                    "type": "quality",
                    "predicted_value": quality_prediction_raw['predicted_class'],
                    "confidence": quality_prediction_raw['confidence'],
                    "score": quality_prediction_raw.get('score'),
                    "input_conditions": input_conditions,
                    "product": product,
                    "batch": batch,
                    "recommendations": quality_prediction_raw.get('recommendations', [])
                }
                all_predictions_data.append(quality_prediction)
                
                # Generate and Save Quality Alerts
                quality_alerts = self._generate_quality_alerts(quality_prediction, product, batch)
                for alert_data in quality_alerts:
                    saved_alert = self._save_alert(alert_data, related_prediction=self._save_prediction(quality_prediction, product, batch))
                    if saved_alert:
                        all_generated_alerts.append(saved_alert)

            except Exception as e:
                logger.error(f"Error during quality prediction for product {product_id}: {e}")

        # Ripeness Prediction
        if 'ripeness' in self.loaded_predictors:
            try:
                ripeness_predictor = self.loaded_predictors['ripeness']
                ripeness_prediction_raw = ripeness_predictor.predict_ripeness(
                    fruit_type=fruit_type, # Added
                    temperature=input_conditions["temperature"],
                    ethylene_level=input_conditions["ethylene_level"],
                    days_since_harvest=days_since_harvest, # Added
                    humidity=input_conditions["humidity"], # Added
                    light_exposure=input_conditions["light_intensity"] # Added
                )
                ripeness_prediction = {
                    "type": "ripeness",
                    "predicted_value": ripeness_prediction_raw['ripeness_stage'],
                    "confidence": ripeness_prediction_raw['ripeness_score'], # Using ripeness_score as confidence for now
                    "score": ripeness_prediction_raw.get('ripeness_score'), # Using ripeness_score as score
                    "ripeness_stage": ripeness_prediction_raw['ripeness_stage'], # Added for alert generation
                    "estimated_days_to_overripe": ripeness_prediction_raw.get('estimated_days_to_overripe'), # Added for alert generation
                    "input_conditions": input_conditions,
                    "product": product,
                    "batch": batch,
                    "recommendations": ripeness_prediction_raw.get('recommendations', [])
                }
                all_predictions_data.append(ripeness_prediction)
                
                # Generate and Save Ripeness Alerts
                ripeness_alerts = self._generate_ripeness_alerts(ripeness_prediction, product, batch)
                for alert_data in ripeness_alerts:
                    saved_alert = self._save_alert(alert_data, related_prediction=self._save_prediction(ripeness_prediction, product, batch))
                    if saved_alert:
                        all_generated_alerts.append(saved_alert)

            except Exception as e:
                logger.error(f"Error during ripeness prediction for product {product_id}: {e}")

        # Disease Prediction
        if 'disease' in self.loaded_predictors:
            try:
                disease_predictor = self.loaded_predictors['disease']
                disease_prediction_raw = disease_predictor.predict_disease(
                    humidity=input_conditions["humidity"],
                    temperature=input_conditions["temperature"] # Assuming temp is also a factor
                )
                disease_prediction = {
                    "type": "disease",
                    "predicted_value": disease_prediction_raw['disease_risk'],
                    "confidence": disease_prediction_raw['confidence'],
                    "score": disease_prediction_raw.get('score'),
                    "input_conditions": input_conditions,
                    "product": product,
                    "batch": batch,
                    "recommendations": disease_prediction_raw.get('recommendations', [])
                }
                all_predictions_data.append(disease_prediction)

                # Generate and Save Disease Alerts
                disease_alerts = self._generate_disease_alerts(disease_prediction, product, batch)
                for alert_data in disease_alerts:
                    saved_alert = self._save_alert(alert_data, related_prediction=self._save_prediction(disease_prediction, product, batch))
                    if saved_alert:
                        all_generated_alerts.append(saved_alert)

            except Exception as e:
                logger.error(f"Error during disease prediction for product {product_id}: {e}")

        # Price Prediction
        if 'price' in self.loaded_predictors:
            try:
                price_predictor = self.loaded_predictors['price']
                price_prediction_raw = price_predictor.predict_price(
                    product_features=product.get_price_features(), # Assuming product has a method to get features
                    market_data=self.get_market_data(product.fruit_type) # Assuming get_market_data method
                )
                price_prediction = {
                    "type": "price",
                    "predicted_value": price_prediction_raw['recommended_price'],
                    "confidence": price_prediction_raw.get('confidence'),
                    "score": price_prediction_raw.get('score'),
                    "input_conditions": input_conditions, # Include for context
                    "product": product,
                    "batch": batch,
                    "recommendations": price_prediction_raw.get('recommendations', [])
                }
                all_predictions_data.append(price_prediction)
                # saved_prediction = self._save_prediction(price_prediction, product, batch)

            except Exception as e:
                logger.error(f"Error during price prediction for product {product_id}: {e}")

        # Shelf Life Prediction
        if 'shelf_life' in self.loaded_predictors:
            try:
                shelf_life_predictor = self.loaded_predictors['shelf_life']
                shelf_life_prediction_raw = shelf_life_predictor.predict_shelf_life(
                    product=product,
                    sensor_data=input_conditions
                )
                shelf_life_prediction = {
                    "type": "shelf_life",
                    "predicted_value": shelf_life_prediction_raw['predicted_value'],
                    "confidence": shelf_life_prediction_raw['confidence'],
                    "score": shelf_life_prediction_raw.get('predicted_days_remaining'), # Using days remaining as score
                    "input_conditions": input_conditions,
                    "product": product,
                    "batch": batch,
                    "recommendations": shelf_life_prediction_raw.get('recommendations', [])
                }
                all_predictions_data.append(shelf_life_prediction)

                # Generate and Save Shelf Life Alerts
                shelf_life_alerts = self._generate_shelf_life_alerts(shelf_life_prediction, product, batch)
                for alert_data in shelf_life_alerts:
                    saved_alert = self._save_alert(alert_data, related_prediction=self._save_prediction(shelf_life_prediction, product, batch))
                    if saved_alert:
                        all_generated_alerts.append(saved_alert)

            except Exception as e:
                logger.error(f"Error during shelf life prediction for product {product_id}: {e}")

        # Ethylene Risk Prediction
        if 'ethylene' in self.loaded_predictors:
            try:
                ethylene_predictor = self.loaded_predictors['ethylene']
                # For this to work, we need `product` to have a `fruit_type` attribute
                # and ideally some way to know other 'producers_in_area' or assume for now.
                # Assuming volume_m3 and ventilation_rate are default or product-specific
                
                # Retrieve fruit_type from product model (assuming it exists)
                fruit_type = product.fruit_type.name if hasattr(product.fruit_type, 'name') else product.name
                
                # Placeholder for producers_in_area for now, ideally derived from inventory
                producers_in_area = [] 
                # Could be improved by checking other products in the same StorageLocation

                ethylene_prediction_raw = ethylene_predictor.predict_ethylene_risk(
                    fruit_type=fruit_type,
                    ethylene_level=input_conditions.get('ethylene_level', 0.0),
                    producers_in_area=producers_in_area,
                    volume_m3=getattr(product.storage_location, 'volume_m3', 100.0) if hasattr(product, 'storage_location') else 100.0,
                    ventilation_rate=getattr(product.storage_location, 'ventilation_rate', 1.0) if hasattr(product, 'storage_location') else 1.0,
                )
                ethylene_prediction = {
                    "type": "ethylene",
                    "predicted_value": ethylene_prediction_raw['predicted_value'],
                    "risk_level": ethylene_prediction_raw['risk_level'],
                    "confidence": ethylene_prediction_raw['confidence'],
                    "score": ethylene_prediction_raw['ethylene_concentration_ppm'], # Using ppm as score
                    "input_conditions": input_conditions,
                    "product": product,
                    "batch": batch,
                    "recommendations": ethylene_prediction_raw.get('recommendations', [])
                }
                all_predictions_data.append(ethylene_prediction)

                # Generate and Save Ethylene Alerts
                ethylene_alerts = self._generate_ethylene_alerts(ethylene_prediction, product, batch)
                for alert_data in ethylene_alerts:
                    saved_alert = self._save_alert(alert_data, related_prediction=self._save_prediction(ethylene_prediction, product, batch))
                    if saved_alert:
                        all_generated_alerts.append(saved_alert)

            except Exception as e:
                logger.error(f"Error during ethylene prediction for product {product_id}: {e}")

        # 6. Send Notifications
        if all_generated_alerts:
            self._send_alert_notifications(all_generated_alerts)

        return {
            "product_id": product_id,
            "batch_id": batch_id,
            "sensor_data_used": input_conditions,
            "predictions": all_predictions_data,
            "alerts_generated": [alert.title for alert in all_generated_alerts]
        }

    def generate_product_insight_report(self, product_id: int, days_back: int = 7) -> Dict[str, Any]:
        """
        Generates a summary report of recent predictions and alerts for a given product.
        """
        try:
            product = Product.objects.get(id=product_id)
        except Product.DoesNotExist:
            logger.error(f"Product with ID {product_id} not found for report generation.")
            return {"error": f"Product with ID {product_id} not found."}

        start_date = timezone.now() - timedelta(days=days_back)
        report = {
            "product_id": product_id,
            "product_name": product.name,
            "report_period_days": days_back,
            "generated_at": timezone.now().isoformat(),
            "predictions_summary": {},
            "alerts_summary": {},
            "overall_status": "OK",
            "recommendations": []
        }

        # --- Summarize Predictions ---
        predictions = FruitPrediction.objects.filter(
            product=product,
            recorded_at__gte=start_date
        ).order_by('-recorded_at')

        report['predictions_summary']['total_predictions_count'] = predictions.count()
        report['predictions_summary']['predictions_by_type'] = {}
        report['predictions_summary']['latest_predictions'] = {}

        prediction_types = predictions.values_list('prediction_type', flat=True).distinct()
        for p_type in prediction_types:
            type_predictions = predictions.filter(prediction_type=p_type)
            if type_predictions.exists():
                latest_pred = type_predictions.first()
                report['predictions_summary']['predictions_by_type'][p_type] = {
                    "count": type_predictions.count(),
                    "avg_confidence": type_predictions.aggregate(Avg('confidence'))['confidence__avg'] or 0.0,
                    "avg_score": type_predictions.aggregate(Avg('predicted_score'))['predicted_score__avg'] or 0.0,
                }
                report['predictions_summary']['latest_predictions'][p_type] = {
                    "value": latest_pred.predicted_value,
                    "confidence": latest_pred.confidence,
                    "score": latest_pred.predicted_score,
                    "alert_level": latest_pred.alert_level,
                    "recorded_at": latest_pred.recorded_at.isoformat()
                }
                if latest_pred.alert_level in ['critical', 'warning']:
                    report['recommendations'].append(f"Based on latest {p_type} prediction: {latest_pred.alert_message}")

        # --- Summarize Alerts ---
        alerts = AlertNotification.objects.filter(
            product=product,
            created_at__gte=start_date
        ).order_by('-created_at')

        report['alerts_summary']['total_alerts_count'] = alerts.count()
        report['alerts_summary']['alerts_by_priority'] = {}
        report['alerts_summary']['recent_critical_alerts'] = []

        alert_priorities = alerts.values_list('priority', flat=True).distinct()
        for ap in alert_priorities:
            report['alerts_summary']['alerts_by_priority'][ap] = alerts.filter(priority=ap).count()
        
        critical_alerts = alerts.filter(priority__in=['critical', 'high'])[:5] # Get up to 5 recent critical alerts
        for alert in critical_alerts:
            report['alerts_summary']['recent_critical_alerts'].append({
                "title": alert.title,
                "message": alert.message,
                "priority": alert.priority,
                "created_at": alert.created_at.isoformat()
            })
            report['recommendations'].append(f"URGENT ALERT: {alert.title} - {alert.message}")
            if alert.priority == 'critical':
                report['overall_status'] = "CRITICAL"
            elif alert.priority == 'high' and report['overall_status'] != "CRITICAL":
                report['overall_status'] = "WARNING"


        # --- Determine Overall Status ---
        # If overall_status is still OK, check latest predictions for warnings
        if report['overall_status'] == "OK":
            for p_type, pred_data in report['predictions_summary']['latest_predictions'].items():
                if pred_data['alert_level'] == 'critical':
                    report['overall_status'] = "CRITICAL"
                    break
                elif pred_data['alert_level'] == 'warning':
                    report['overall_status'] = "WARNING"
            
        # Refine recommendations based on overall status
        if report['overall_status'] == "OK" and not report['recommendations']:
            report['recommendations'].append(f"Product {product.name} is stable and within acceptable parameters.")
        elif not report['recommendations'] and report['overall_status'] != "OK":
             report['recommendations'].append(f"Product {product.name} requires attention based on recent insights.")


        return report