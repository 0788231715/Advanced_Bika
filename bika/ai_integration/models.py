# bika/ai_integration/models.py
from django.db import models
from django.utils import timezone
from bika.models import Product, StorageLocation, FruitBatch, TrainedModel



class FruitPrediction(models.Model):
    """Store predictions for each fruit/product"""
    PREDICTION_TYPES = [
        ('quality', 'Quality'),
        ('ripeness', 'Ripeness'),
        ('shelf_life', 'Shelf Life'),
        ('disease', 'Disease Risk'),
        ('price', 'Price'),
    ]
    
    ALERT_LEVELS = [
        ('info', 'Information'),
        ('warning', 'Warning'),
        ('critical', 'Critical'),
        ('emergency', 'Emergency'),
    ]
    
    # Link to your existing product
    product = models.ForeignKey(FruitProduct, on_delete=models.CASCADE, related_name='predictions')
    batch = models.ForeignKey(Batch, on_delete=models.CASCADE, null=True, blank=True)
    
    # Prediction details
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    predicted_value = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)  # 0-1
    predicted_score = models.FloatField(null=True, blank=True)  # Numeric score if applicable
    
    # Sensor data used
    temperature = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    light_intensity = models.FloatField(null=True, blank=True)
    co2_level = models.FloatField(null=True, blank=True)
    ethylene_level = models.FloatField(null=True, blank=True)
    
    # Alert information
    alert_level = models.CharField(max_length=20, choices=ALERT_LEVELS, default='info')
    alert_message = models.TextField()
    recommendations = models.JSONField(default=list)  # List of recommendations
    
    # Metadata
    prediction_date = models.DateTimeField(default=timezone.now)
    model_used = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True)
    is_verified = models.BooleanField(default=False)  # Human verification
    verified_by = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, blank=True)
    verified_date = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-prediction_date']
        indexes = [
            models.Index(fields=['product', 'prediction_date']),
            models.Index(fields=['alert_level', 'prediction_date']),
        ]
    
    def __str__(self):
        return f"{self.product.name} - {self.prediction_type}: {self.predicted_value}"

class AlertNotification(models.Model):
    """Store alerts for immediate action"""
    ALERT_TYPES = [
        ('quality_drop', 'Quality Drop'),
        ('temperature_issue', 'Temperature Issue'),
        ('humidity_issue', 'Humidity Issue'),
        ('disease_risk', 'Disease Risk'),
        ('shelf_life', 'Shelf Life Warning'),
        ('price_change', 'Price Change'),
        ('ethylene_risk', 'Ethylene Risk'),
    ]
    
    PRIORITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPES)
    priority = models.CharField(max_length=20, choices=PRIORITY_LEVELS)
    title = models.CharField(max_length=200)
    message = models.TextField()
    
    # Related items
    product = models.ForeignKey(FruitProduct, on_delete=models.CASCADE, null=True, blank=True)
    batch = models.ForeignKey(Batch, on_delete=models.CASCADE, null=True, blank=True)
    warehouse = models.ForeignKey(Warehouse, on_delete=models.CASCADE, null=True, blank=True)
    prediction = models.ForeignKey(FruitPrediction, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Status tracking
    is_resolved = models.BooleanField(default=False)
    resolved_by = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, blank=True)
    resolved_date = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.ForeignKey('auth.User', on_delete=models.SET_NULL, 
                                        null=True, blank=True, related_name='acknowledged_alerts')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.alert_type} - {self.title} ({self.priority})"