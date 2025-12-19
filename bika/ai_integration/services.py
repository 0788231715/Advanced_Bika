# bika/ai_integration/services.py
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.core.mail import send_mail
from django.contrib.auth.models import User

from bika.models import FruitProduct, Batch, Warehouse, SensorReading
from .models import TrainedModel, FruitPrediction, AlertNotification

class FruitAIService:
    """Main AI service for fruit quality monitoring"""
    
    def __init__(self):
        self.models = {}
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load all active trained models"""
        active_models = TrainedModel.objects.filter(is_active=True)
        
        for model_record in active_models:
            try:
                if os.path.exists(model_record.model_file.path):
                    model_data = joblib.load(model_record.model_file.path)
                    self.models[model_record.model_type] = {
                        'model': model_data['model'],
                        'scaler': model_data.get('scaler'),
                        'label_encoder': model_data.get('label_encoder'),
                        'metadata': model_data.get('metadata', {}),
                        'record': model_record
                    }
                    print(f"✅ Loaded {model_record.model_type} model: {model_record.name}")
            except Exception as e:
                print(f"❌ Error loading model {model_record.name}: {e}")
    
    def predict_product_quality(self, product_id, sensor_data=None):
        """
        Predict quality for a specific product
        Returns: Prediction object with alerts
        """
        try:
            # Get product
            product = FruitProduct.objects.get(id=product_id)
            
            # Get latest sensor data if not provided
            if sensor_data is None:
                sensor_data = self._get_latest_sensor_data(product)
            
            # Get batch if available
            batch = product.batches.filter(status='active').first()
            
            # Make predictions
            predictions = []
            alerts = []
            
            # 1. Quality prediction
            if 'quality' in self.models:
                quality_pred = self._predict_quality(product, sensor_data)
                predictions.append(quality_pred)
                
                # Generate quality alerts
                quality_alerts = self._generate_quality_alerts(quality_pred, product, batch)
                alerts.extend(quality_alerts)
            
            # 2. Ripeness prediction
            if 'ripeness' in self.models:
                ripeness_pred = self._predict_ripeness(product, sensor_data)
                predictions.append(ripeness_pred)
            
            # 3. Disease risk
            if 'disease' in self.models:
                disease_pred = self._predict_disease_risk(product, sensor_data)
                predictions.append(disease_pred)
                
                # Generate disease alerts
                disease_alerts = self._generate_disease_alerts(disease_pred, product, batch)
                alerts.extend(disease_alerts)
            
            # 4. Shelf life estimation
            shelf_life_pred = self._estimate_shelf_life(product, sensor_data)
            predictions.append(shelf_life_pred)
            
            # 5. Price prediction
            if 'price' in self.models:
                price_pred = self._predict_price(product, sensor_data)
                predictions.append(price_pred)
            
            # Save all predictions
            saved_predictions = []
            for pred in predictions:
                saved_pred = self._save_prediction(pred)
                if saved_pred:
                    saved_predictions.append(saved_pred)
            
            # Save and send alerts
            saved_alerts = []
            for alert in alerts:
                saved_alert = self._save_alert(alert, saved_predictions)
                if saved_alert:
                    saved_alerts.append(saved_alert)
            
            # Send email notifications for critical alerts
            self._send_alert_notifications(saved_alerts)
            
            return {
                'success': True,
                'predictions': saved_predictions,
                'alerts': saved_alerts,
                'product': product.name,
                'timestamp': timezone.now()
            }
            
        except Exception as e:
            print(f"❌ Error predicting quality: {e}")
            return {'error': str(e)}
    
    def _get_latest_sensor_data(self, product):
        """Get latest sensor readings for a product"""
        # Try to get from product's warehouse
        warehouse = product.warehouse
        if warehouse:
            # Get latest sensor readings from warehouse
            readings = SensorReading.objects.filter(
                warehouse=warehouse,
                timestamp__gte=timezone.now() - timedelta(hours=24)
            ).order_by('-timestamp')
            
            if readings.exists():
                latest = readings.first()
                return {
                    'temperature': latest.temperature,
                    'humidity': latest.humidity,
                    'light_intensity': latest.light_intensity,
                    'co2_level': latest.co2_level,
                    'ethylene_level': latest.ethylene_level
                }
        
        # Fallback to product's own sensor data
        return {
            'temperature': product.current_temperature or 22.0,
            'humidity': product.current_humidity or 85.0,
            'light_intensity': product.light_exposure or 50.0,
            'co2_level': product.co2_level or 400.0,
            'ethylene_level': product.ethylene_level or 0.0
        }
    
    def _predict_quality(self, product, sensor_data):
        """Predict fruit quality"""
        model_info = self.models.get('quality')
        if not model_info:
            return None
        
        try:
            # Prepare input data
            input_features = self._prepare_features(product, sensor_data, model_info)
            
            # Make prediction
            model = model_info['model']
            scaler = model_info['scaler']
            le = model_info['label_encoder']
            
            # Scale features if scaler exists
            if scaler:
                input_scaled = scaler.transform([input_features])
            else:
                input_scaled = [input_features]
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Decode if label encoder exists
            if le:
                predicted_class = le.inverse_transform([prediction])[0]
            else:
                predicted_class = str(prediction)
            
            # Get confidence
            confidence = prediction_proba.max() if prediction_proba is not None else 0.8
            
            return {
                'type': 'quality',
                'predicted_value': predicted_class,
                'confidence': float(confidence),
                'score': float(prediction_proba.max()) if prediction_proba is not None else 0.8,
                'sensor_data': sensor_data,
                'product': product,
                'model_used': model_info['record']
            }
            
        except Exception as e:
            print(f"Error in quality prediction: {e}")
            return None
    
    def _predict_ripeness(self, product, sensor_data):
        """Predict fruit ripeness"""
        # Simple ripeness calculation based on time and conditions
        days_since_harvest = (timezone.now() - product.harvest_date).days if product.harvest_date else 7
        
        ripeness_score = min(1.0, days_since_harvest / 10)
        
        if ripeness_score < 0.3:
            stage = 'unripe'
        elif ripeness_score < 0.6:
            stage = 'ripe'
        elif ripeness_score < 0.8:
            stage = 'fully_ripe'
        else:
            stage = 'overripe'
        
        return {
            'type': 'ripeness',
            'predicted_value': stage,
            'confidence': 0.7,
            'score': ripeness_score,
            'sensor_data': sensor_data,
            'product': product
        }
    
    def _predict_disease_risk(self, product, sensor_data):
        """Predict disease risk"""
        # Simple disease risk calculation
        risk_score = 0.0
        
        # Temperature effect
        if sensor_data['temperature'] > 25:
            risk_score += 0.3
        elif sensor_data['temperature'] < 5:
            risk_score += 0.2
        
        # Humidity effect
        if sensor_data['humidity'] > 95:
            risk_score += 0.4
        elif sensor_data['humidity'] < 70:
            risk_score += 0.1
        
        # CO2 effect
        if sensor_data['co2_level'] > 1000:
            risk_score += 0.2
        
        risk_score = min(1.0, risk_score)
        
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'type': 'disease',
            'predicted_value': risk_level,
            'confidence': 0.8,
            'score': risk_score,
            'sensor_data': sensor_data,
            'product': product
        }
    
    def _estimate_shelf_life(self, product, sensor_data):
        """Estimate remaining shelf life"""
        # Base shelf life by fruit type
        base_shelf_life = {
            'banana': 7,
            'apple': 30,
            'orange': 21,
            'mango': 10,
            'tomato': 14,
            'strawberry': 5,
        }.get(product.fruit_type.lower(), 10)
        
        # Adjust based on conditions
        temp_factor = 1.0
        if sensor_data['temperature'] > 15:
            temp_factor = 0.5
        elif sensor_data['temperature'] > 10:
            temp_factor = 0.7
        
        humidity_factor = 1.0
        if sensor_data['humidity'] < 80:
            humidity_factor = 0.7
        
        estimated_days = base_shelf_life * temp_factor * humidity_factor
        
        # Calculate days remaining
        if product.expiry_date:
            days_remaining = (product.expiry_date - timezone.now().date()).days
        else:
            days_remaining = max(0, int(estimated_days))
        
        if days_remaining <= 2:
            status = 'critical'
        elif days_remaining <= 5:
            status = 'warning'
        else:
            status = 'good'
        
        return {
            'type': 'shelf_life',
            'predicted_value': f"{days_remaining} days",
            'confidence': 0.8,
            'score': days_remaining,
            'sensor_data': sensor_data,
            'product': product,
            'status': status
        }
    
    def _predict_price(self, product, sensor_data):
        """Predict price based on quality"""
        quality_map = {
            'Excellent': 1.2,
            'Good': 1.0,
            'Fair': 0.8,
            'Poor': 0.5,
            'Bad': 0.2
        }
        
        # Get base price
        base_price = product.original_price or 1000
        
        # Adjust based on estimated quality
        if sensor_data['temperature'] > 20 and sensor_data['humidity'] > 90:
            quality_factor = 0.7
        elif sensor_data['temperature'] < 15 and sensor_data['humidity'] > 85:
            quality_factor = 1.0
        else:
            quality_factor = 0.9
        
        predicted_price = base_price * quality_factor
        
        return {
            'type': 'price',
            'predicted_value': f"₹{predicted_price:.2f}",
            'confidence': 0.6,
            'score': predicted_price,
            'sensor_data': sensor_data,
            'product': product
        }
    
    def _prepare_features(self, product, sensor_data, model_info):
        """Prepare features for model prediction"""
        # Get feature columns from model metadata
        feature_columns = model_info['metadata'].get('feature_columns', [])
        
        # Map product data to features
        features = []
        for col in feature_columns:
            if col in sensor_data:
                features.append(sensor_data[col])
            elif hasattr(product, col):
                features.append(getattr(product, col))
            elif col == 'fruit_type':
                features.append(product.fruit_type)
            elif col == 'days_since_harvest':
                if product.harvest_date:
                    days = (timezone.now() - product.harvest_date).days
                    features.append(days)
                else:
                    features.append(7)  # Default
            else:
                features.append(0.0)  # Default value
        
        return features
    
    def _generate_quality_alerts(self, prediction, product, batch):
        """Generate alerts based on quality prediction"""
        alerts = []
        
        if not prediction:
            return alerts
        
        quality = prediction['predicted_value']
        confidence = prediction['confidence']
        
        # Quality-based alerts
        if quality in ['Bad', 'Poor', 'Rotten']:
            alerts.append({
                'type': 'quality_drop',
                'priority': 'critical' if quality == 'Rotten' else 'high',
                'title': f'{product.name} Quality Alert: {quality}',
                'message': f'Product "{product.name}" is predicted as {quality} quality with {confidence:.0%} confidence.',
                'product': product,
                'batch': batch,
                'recommendations': [
                    'Move to discount section',
                    'Check for immediate processing',
                    'Isolate from other products'
                ]
            })
        
        # Temperature alert
        sensor_data = prediction.get('sensor_data', {})
        if sensor_data.get('temperature', 22) > 25:
            alerts.append({
                'type': 'temperature_issue',
                'priority': 'high',
                'title': f'High Temperature Alert for {product.name}',
                'message': f'Temperature ({sensor_data["temperature"]}°C) is above optimal range.',
                'product': product,
                'batch': batch,
                'recommendations': [
                    'Adjust cooling system',
                    'Move to cooler area',
                    'Check ventilation'
                ]
            })
        
        return alerts
    
    def _generate_disease_alerts(self, prediction, product, batch):
        """Generate disease risk alerts"""
        alerts = []
        
        if not prediction:
            return alerts
        
        risk_level = prediction['predicted_value']
        risk_score = prediction['score']
        
        if risk_level in ['high', 'medium']:
            alerts.append({
                'type': 'disease_risk',
                'priority': 'high' if risk_level == 'high' else 'medium',
                'title': f'Disease Risk Alert: {risk_level.title()} Risk',
                'message': f'{product.name} has {risk_level} disease risk ({risk_score:.0%}).',
                'product': product,
                'batch': batch,
                'recommendations': [
                    'Increase ventilation',
                    'Check for mold/spoilage',
                    'Consider fungicide treatment',
                    'Monitor closely'
                ]
            })
        
        return alerts
    
    def _save_prediction(self, prediction_data):
        """Save prediction to database"""
        if not prediction_data:
            return None
        
        try:
            with transaction.atomic():
                prediction = FruitPrediction(
                    product=prediction_data['product'],
                    batch=prediction_data.get('batch'),
                    prediction_type=prediction_data['type'],
                    predicted_value=str(prediction_data['predicted_value']),
                    confidence=prediction_data['confidence'],
                    predicted_score=prediction_data.get('score'),
                    
                    # Sensor data
                    temperature=prediction_data['sensor_data'].get('temperature'),
                    humidity=prediction_data['sensor_data'].get('humidity'),
                    light_intensity=prediction_data['sensor_data'].get('light_intensity'),
                    co2_level=prediction_data['sensor_data'].get('co2_level'),
                    ethylene_level=prediction_data['sensor_data'].get('ethylene_level'),
                    
                    # Alert info (determine based on prediction)
                    alert_level=self._determine_alert_level(prediction_data),
                    alert_message=self._generate_alert_message(prediction_data),
                    recommendations=self._generate_recommendations(prediction_data),
                    
                    model_used=prediction_data.get('model_used')
                )
                prediction.save()
                return prediction
                
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return None
    
    def _determine_alert_level(self, prediction_data):
        """Determine alert level based on prediction"""
        pred_type = prediction_data['type']
        pred_value = str(prediction_data['predicted_value']).lower()
        score = prediction_data.get('score', 0)
        
        if pred_type == 'quality':
            if pred_value in ['rotten', 'bad']:
                return 'critical'
            elif pred_value in ['poor', 'fair']:
                return 'warning'
        
        elif pred_type == 'disease':
            if score > 0.7:
                return 'critical'
            elif score > 0.4:
                return 'warning'
        
        elif pred_type == 'shelf_life':
            if score <= 2:
                return 'critical'
            elif score <= 5:
                return 'warning'
        
        return 'info'
    
    def _generate_alert_message(self, prediction_data):
        """Generate alert message from prediction"""
        pred_type = prediction_data['type']
        pred_value = prediction_data['predicted_value']
        confidence = prediction_data['confidence']
        product = prediction_data['product']
        
        messages = {
            'quality': f'Quality predicted as {pred_value} with {confidence:.0%} confidence.',
            'ripeness': f'Ripeness stage: {pred_value}',
            'disease': f'Disease risk: {pred_value}',
            'shelf_life': f'Remaining shelf life: {pred_value}',
            'price': f'Recommended price: {pred_value}'
        }
        
        return messages.get(pred_type, f'{pred_type}: {pred_value}')
    
    def _generate_recommendations(self, prediction_data):
        """Generate recommendations based on prediction"""
        pred_type = prediction_data['type']
        pred_value = str(prediction_data['predicted_value']).lower()
        
        recommendations = []
        
        if pred_type == 'quality':
            if pred_value in ['rotten', 'bad']:
                recommendations = ['Immediate disposal', 'Check for contamination', 'Document loss']
            elif pred_value == 'poor':
                recommendations = ['Discount sale', 'Process into products', 'Priority sale']
            elif pred_value == 'fair':
                recommendations = ['Standard sale', 'Monitor quality', 'Regular pricing']
        
        elif pred_type == 'disease' and pred_value == 'high':
            recommendations = ['Isolate product', 'Increase ventilation', 'Apply treatment']
        
        elif pred_type == 'shelf_life':
            days = prediction_data.get('score', 0)
            if days <= 2:
                recommendations = ['Emergency sale', 'Immediate processing', 'Document expiry']
            elif days <= 5:
                recommendations = ['Priority sale', 'Discount pricing', 'Monitor closely']
        
        return recommendations
    
    def _save_alert(self, alert_data, related_predictions):
        """Save alert to database"""
        try:
            # Find related prediction
            related_prediction = None
            for pred in related_predictions:
                if pred.product == alert_data.get('product'):
                    related_prediction = pred
                    break
            
            alert = AlertNotification(
                alert_type=alert_data['type'],
                priority=alert_data['priority'],
                title=alert_data['title'],
                message=alert_data['message'],
                product=alert_data.get('product'),
                batch=alert_data.get('batch'),
                prediction=related_prediction
            )
            alert.save()
            return alert
            
        except Exception as e:
            print(f"Error saving alert: {e}")
            return None
    
    def _send_alert_notifications(self, alerts):
        """Send email notifications for critical alerts"""
        critical_alerts = [a for a in alerts if a.priority in ['critical', 'high']]
        
        if not critical_alerts:
            return
        
        try:
            # Get admin users
            admin_users = User.objects.filter(is_staff=True, is_active=True)
            admin_emails = [user.email for user in admin_users if user.email]
            
            if not admin_emails:
                return
            
            # Prepare email content
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
            
            # Send email
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=admin_emails,
                fail_silently=True
            )
            
            print(f"📧 Sent {len(critical_alerts)} critical alerts to {len(admin_emails)} admins")
            
        except Exception as e:
            print(f"Error sending alert emails: {e}")