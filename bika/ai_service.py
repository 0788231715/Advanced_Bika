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

# Import AI models from the main models module
from bika.ai_models import (
    FruitQualityPredictor, FruitRipenessPredictor,
    EthyleneMonitor, FruitDiseasePredictor, FruitPricePredictor,
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
        
    def get_model_performance(self, model_type='quality'):
        """Get performance metrics for trained models"""
        try:
            model_dir = os.path.join(settings.MEDIA_ROOT, 'fruit_models')
            
            if not os.path.exists(model_dir):
                return {'error': 'No models directory found'}
            
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            if not model_files:
                return {'error': 'No trained models found'}
            
            performance_data = []
            
            for model_file in model_files:
                try:
                    model_path = os.path.join(model_dir, model_file)
                    
                    # Try to load model metadata without loading full model
                    if hasattr(self.quality_predictor, 'load_model_metadata'):
                        metadata = self.quality_predictor.load_model_metadata(model_path)
                    else:
                        # Fallback: load model and get metrics
                        temp_predictor = FruitQualityPredictor()
                        if temp_predictor.load_model(model_path):
                            metadata = {
                                'file_name': model_file,
                                'model_type': temp_predictor.model_type,
                                'accuracy': temp_predictor.model_metrics.get('accuracy', 0),
                                'training_samples': temp_predictor.model_metrics.get('training_samples', 0),
                                'created_at': os.path.getctime(model_path)
                            }
                        else:
                            metadata = None
                    
                    if metadata:
                        performance_data.append(metadata)
                        
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")
                    continue
            
            # Sort by accuracy (descending)
            performance_data.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
            
            return {
                'total_models': len(performance_data),
                'best_model': performance_data[0] if performance_data else None,
                'all_models': performance_data,
                'average_accuracy': np.mean([m.get('accuracy', 0) for m in performance_data]) if performance_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {'error': str(e)}
    
    def validate_dataset(self, csv_file):
        """Validate dataset before training"""
        try:
            # Save file temporarily
            timestamp = int(timezone.now().timestamp())
            temp_path = os.path.join('temp_datasets', f'validate_{timestamp}.csv')
            saved_path = default_storage.save(temp_path, csv_file)
            full_path = default_storage.path(saved_path)
            
            # Load dataset
            df = pd.read_csv(full_path)
            
            # Perform validation checks
            validation_results = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'columns': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
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
    
    def batch_predict(self, predictions_data):
        """Make predictions for multiple data points"""
        try:
            results = []
            
            for data in predictions_data:
                prediction = self.predict_fruit_quality(
                    data.get('fruit_name'),
                    data.get('temperature'),
                    data.get('humidity'),
                    data.get('light_intensity'),
                    data.get('co2_level'),
                    data.get('batch_id')
                )
                
                if prediction.get('success'):
                    results.append({
                        'input_data': data,
                        'prediction': prediction,
                        'timestamp': timezone.now().isoformat()
                    })
            
            # Calculate batch statistics
            if results:
                quality_classes = [r['prediction']['quality_prediction']['predicted_class'] for r in results]
                quality_counts = pd.Series(quality_classes).value_counts().to_dict()
                
                confidence_scores = [r['prediction']['quality_prediction']['confidence'] for r in results]
                
                batch_stats = {
                    'total_predictions': len(results),
                    'quality_distribution': quality_counts,
                    'avg_confidence': np.mean(confidence_scores),
                    'min_confidence': np.min(confidence_scores),
                    'max_confidence': np.max(confidence_scores),
                    'most_common_quality': max(quality_counts.items(), key=lambda x: x[1])[0] if quality_counts else None
                }
                
                # Batch recommendations
                batch_recommendations = []
                if 'Rotten' in quality_counts and quality_counts['Rotten'] / len(results) > 0.3:
                    batch_recommendations.append("High proportion of rotten predictions - immediate action required")
                
                if np.mean(confidence_scores) < 0.6:
                    batch_recommendations.append("Low average confidence - consider model retraining")
                
                batch_stats['recommendations'] = batch_recommendations
            else:
                batch_stats = {}
            
            return {
                'success': True,
                'individual_results': results,
                'batch_statistics': batch_stats,
                'total_processed': len(predictions_data)
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {'error': str(e)}
    
    def optimize_storage_conditions(self, fruit_type, current_conditions, target_quality='Good'):
        """Optimize storage conditions for target quality"""
        try:
            # Get fruit type info from database
            from bika.models import FruitType
            try:
                fruit_info = FruitType.objects.get(name__icontains=fruit_type)
                optimal_temp_range = (float(fruit_info.optimal_temp_min), float(fruit_info.optimal_temp_max))
                optimal_humidity_range = (float(fruit_info.optimal_humidity_min), float(fruit_info.optimal_humidity_max))
                optimal_light_max = fruit_info.optimal_light_max
                optimal_co2_max = fruit_info.optimal_co2_max
            except:
                # Use default values if fruit type not found
                optimal_temp_range = (2.0, 8.0)
                optimal_humidity_range = (85.0, 95.0)
                optimal_light_max = 100
                optimal_co2_max = 400
            
            current_temp = current_conditions.get('temperature', 5.0)
            current_humidity = current_conditions.get('humidity', 90.0)
            current_light = current_conditions.get('light_intensity', 50.0)
            current_co2 = current_conditions.get('co2_level', 400.0)
            
            # Calculate adjustments needed
            adjustments = []
            
            # Temperature adjustment
            if current_temp < optimal_temp_range[0]:
                adjustments.append({
                    'parameter': 'temperature',
                    'current': current_temp,
                    'optimal_min': optimal_temp_range[0],
                    'optimal_max': optimal_temp_range[1],
                    'adjustment': f"Increase to {optimal_temp_range[0]} - {optimal_temp_range[1]}°C",
                    'priority': 'high'
                })
            elif current_temp > optimal_temp_range[1]:
                adjustments.append({
                    'parameter': 'temperature',
                    'current': current_temp,
                    'optimal_min': optimal_temp_range[0],
                    'optimal_max': optimal_temp_range[1],
                    'adjustment': f"Decrease to {optimal_temp_range[0]} - {optimal_temp_range[1]}°C",
                    'priority': 'high'
                })
            
            # Humidity adjustment
            if current_humidity < optimal_humidity_range[0]:
                adjustments.append({
                    'parameter': 'humidity',
                    'current': current_humidity,
                    'optimal_min': optimal_humidity_range[0],
                    'optimal_max': optimal_humidity_range[1],
                    'adjustment': f"Increase to {optimal_humidity_range[0]} - {optimal_humidity_range[1]}%",
                    'priority': 'medium'
                })
            elif current_humidity > optimal_humidity_range[1]:
                adjustments.append({
                    'parameter': 'humidity',
                    'current': current_humidity,
                    'optimal_min': optimal_humidity_range[0],
                    'optimal_max': optimal_humidity_range[1],
                    'adjustment': f"Decrease to {optimal_humidity_range[0]} - {optimal_humidity_range[1]}%",
                    'priority': 'medium'
                })
            
            # Light adjustment
            if current_light > optimal_light_max:
                adjustments.append({
                    'parameter': 'light_intensity',
                    'current': current_light,
                    'optimal_max': optimal_light_max,
                    'adjustment': f"Reduce to below {optimal_light_max} lux",
                    'priority': 'low'
                })
            
            # CO2 adjustment
            if current_co2 > optimal_co2_max:
                adjustments.append({
                    'parameter': 'co2_level',
                    'current': current_co2,
                    'optimal_max': optimal_co2_max,
                    'adjustment': f"Improve ventilation to reduce below {optimal_co2_max} ppm",
                    'priority': 'medium'
                })
            
            # Predict quality with optimized conditions
            optimized_temp = min(max(current_temp, optimal_temp_range[0]), optimal_temp_range[1])
            optimized_humidity = min(max(current_humidity, optimal_humidity_range[0]), optimal_humidity_range[1])
            optimized_light = min(current_light, optimal_light_max)
            optimized_co2 = min(current_co2, optimal_co2_max)
            
            current_prediction = self.predict_fruit_quality(
                fruit_type, current_temp, current_humidity, current_light, current_co2
            )
            
            optimized_prediction = self.predict_fruit_quality(
                fruit_type, optimized_temp, optimized_humidity, optimized_light, optimized_co2
            )
            
            # Calculate improvement
            quality_scores = {'Fresh': 100, 'Good': 80, 'Fair': 60, 'Poor': 30, 'Rotten': 0}
            current_score = quality_scores.get(
                current_prediction.get('quality_prediction', {}).get('predicted_class', 'Unknown'), 
                50
            )
            optimized_score = quality_scores.get(
                optimized_prediction.get('quality_prediction', {}).get('predicted_class', 'Unknown'), 
                50
            )
            
            improvement = optimized_score - current_score
            
            return {
                'success': True,
                'fruit_type': fruit_type,
                'current_conditions': current_conditions,
                'optimal_ranges': {
                    'temperature': optimal_temp_range,
                    'humidity': optimal_humidity_range,
                    'light_intensity': optimal_light_max,
                    'co2_level': optimal_co2_max
                },
                'adjustments_needed': adjustments,
                'current_prediction': current_prediction,
                'optimized_prediction': optimized_prediction,
                'quality_improvement': improvement,
                'estimated_shelf_life_improvement': self._estimate_shelf_life_improvement(improvement),
                'recommendations': self._generate_optimization_recommendations(adjustments, improvement)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing storage conditions: {e}")
            return {'error': str(e)}
    
    def _estimate_shelf_life_improvement(self, quality_improvement):
        """Estimate shelf life improvement based on quality improvement"""
        # Simple linear relationship: 10 quality points ≈ 1 day shelf life
        return max(0, quality_improvement / 10)
    
    def _generate_optimization_recommendations(self, adjustments, improvement):
        """Generate recommendations based on optimization results"""
        recommendations = []
        
        if not adjustments:
            recommendations.append("Current conditions are optimal. Maintain current settings.")
        else:
            # Priority-based recommendations
            high_priority = [a for a in adjustments if a['priority'] == 'high']
            medium_priority = [a for a in adjustments if a['priority'] == 'medium']
            low_priority = [a for a in adjustments if a['priority'] == 'low']
            
            if high_priority:
                recommendations.append("High priority adjustments needed:")
                for adj in high_priority:
                    recommendations.append(f"- {adj['parameter']}: {adj['adjustment']}")
            
            if medium_priority:
                recommendations.append("Medium priority adjustments:")
                for adj in medium_priority:
                    recommendations.append(f"- {adj['parameter']}: {adj['adjustment']}")
            
            if low_priority:
                recommendations.append("Low priority adjustments (if possible):")
                for adj in low_priority:
                    recommendations.append(f"- {adj['parameter']}: {adj['adjustment']}")
        
        if improvement > 20:
            recommendations.append(f"Significant quality improvement ({improvement} points) possible with adjustments.")
        elif improvement > 10:
            recommendations.append(f"Moderate quality improvement ({improvement} points) possible.")
        elif improvement > 0:
            recommendations.append(f"Minor quality improvement ({improvement} points) possible.")
        else:
            recommendations.append("No quality improvement expected with current adjustments.")
        
        return recommendations
    
    def generate_quality_report(self, batch_id, start_date=None, end_date=None):
        """Generate comprehensive quality report for a batch"""
        try:
            from bika.models import FruitBatch, FruitQualityReading
            
            batch = FruitBatch.objects.get(id=batch_id)
            
            # Set date range
            if not start_date:
                start_date = timezone.now() - timedelta(days=7)
            if not end_date:
                end_date = timezone.now()
            
            # Get readings
            readings = FruitQualityReading.objects.filter(
                fruit_batch=batch,
                timestamp__range=[start_date, end_date]
            ).order_by('timestamp')
            
            if not readings.exists():
                return {'error': 'No quality readings found in the specified period'}
            
            # Convert to DataFrame
            data = []
            for reading in readings:
                data.append({
                    'timestamp': reading.timestamp,
                    'temperature': float(reading.temperature),
                    'humidity': float(reading.humidity),
                    'light_intensity': float(reading.light_intensity),
                    'co2_level': reading.co2_level,
                    'predicted_class': reading.predicted_class,
                    'confidence': float(reading.confidence_score),
                    'actual_class': reading.actual_class if reading.actual_class else None
                })
            
            df = pd.DataFrame(data)
            
            # Calculate statistics
            stats = {
                'total_readings': len(df),
                'period': f"{start_date.date()} to {end_date.date()}",
                'temperature_stats': {
                    'mean': float(df['temperature'].mean()),
                    'std': float(df['temperature'].std()),
                    'min': float(df['temperature'].min()),
                    'max': float(df['temperature'].max()),
                    'stability': 'Stable' if df['temperature'].std() < 2 else 'Unstable'
                },
                'humidity_stats': {
                    'mean': float(df['humidity'].mean()),
                    'std': float(df['humidity'].std()),
                    'min': float(df['humidity'].min()),
                    'max': float(df['humidity'].max()),
                    'stability': 'Stable' if df['humidity'].std() < 5 else 'Unstable'
                },
                'quality_distribution': df['predicted_class'].value_counts().to_dict(),
                'average_confidence': float(df['confidence'].mean()),
                'quality_trend': self._calculate_quality_trend(df),
                'anomalies': self._detect_anomalies(df)
            }
            
            # Model accuracy (if actual classes are available)
            if df['actual_class'].notna().any():
                actuals = df['actual_class'].dropna()
                predictions = df.loc[actuals.index, 'predicted_class']
                accuracy = (actuals == predictions).sum() / len(actuals)
                stats['model_accuracy'] = float(accuracy)
                stats['misclassified'] = (actuals != predictions).sum()
            
            # Generate insights
            insights = self._generate_quality_insights(df, batch)
            
            # Recommendations
            recommendations = self._generate_report_recommendations(stats, insights, batch)
            
            # Export data (for download)
            export_data = {
                'batch_info': {
                    'batch_number': batch.batch_number,
                    'fruit_type': batch.fruit_type.name,
                    'quantity': batch.quantity,
                    'arrival_date': batch.arrival_date.isoformat(),
                    'expected_expiry': batch.expected_expiry.isoformat() if batch.expected_expiry else None,
                    'days_remaining': batch.days_remaining
                },
                'report_period': stats['period'],
                'statistics': stats,
                'insights': insights,
                'recommendations': recommendations,
                'raw_data_summary': data[-10:] if len(data) > 10 else data,  # Last 10 readings
                'generated_at': timezone.now().isoformat()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {'error': str(e)}
    
    def _calculate_quality_trend(self, df):
        """Calculate quality trend over time"""
        if len(df) < 2:
            return 'insufficient_data'
        
        quality_scores = {'Fresh': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Rotten': 1}
        df['quality_score'] = df['predicted_class'].map(quality_scores)
        
        # Simple linear trend
        x = np.arange(len(df))
        y = df['quality_score'].values
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _detect_anomalies(self, df):
        """Detect anomalies in sensor readings"""
        anomalies = []
        
        # Temperature anomalies
        temp_std = df['temperature'].std()
        temp_mean = df['temperature'].mean()
        temp_anomalies = df[np.abs(df['temperature'] - temp_mean) > 2 * temp_std]
        
        if len(temp_anomalies) > 0:
            anomalies.append({
                'type': 'temperature_anomaly',
                'count': len(temp_anomalies),
                'description': f"Temperature spikes/drops detected ({len(temp_anomalies)} occurrences)"
            })
        
        # Sudden quality drops
        if len(df) >= 3:
            quality_scores = {'Fresh': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Rotten': 1}
            df['quality_score'] = df['predicted_class'].map(quality_scores)
            
            for i in range(1, len(df)):
                if df.iloc[i]['quality_score'] < df.iloc[i-1]['quality_score'] - 1:
                    anomalies.append({
                        'type': 'quality_drop',
                        'timestamp': df.iloc[i]['timestamp'].isoformat(),
                        'from': df.iloc[i-1]['predicted_class'],
                        'to': df.iloc[i]['predicted_class'],
                        'description': f"Sudden quality drop from {df.iloc[i-1]['predicted_class']} to {df.iloc[i]['predicted_class']}"
                    })
        
        return anomalies
    
    def _generate_quality_insights(self, df, batch):
        """Generate insights from quality data"""
        insights = []
        
        # Temperature insights
        temp_mean = df['temperature'].mean()
        if temp_mean < 2:
            insights.append("Temperature consistently too low - risk of chilling injury")
        elif temp_mean > 12:
            insights.append("Temperature consistently too high - accelerated spoilage likely")
        
        # Humidity insights
        humidity_mean = df['humidity'].mean()
        if humidity_mean < 85:
            insights.append("Low humidity detected - dehydration risk")
        elif humidity_mean > 95:
            insights.append("High humidity detected - mold growth risk")
        
        # Quality trend insights
        quality_trend = self._calculate_quality_trend(df)
        if quality_trend == 'deteriorating':
            insights.append("Quality is deteriorating over time - consider priority handling")
        elif quality_trend == 'improving':
            insights.append("Quality is improving - storage conditions may be optimal")
        
        # Confidence insights
        avg_confidence = df['confidence'].mean()
        if avg_confidence < 0.6:
            insights.append("Low prediction confidence - model may need retraining")
        
        # Batch-specific insights
        if batch.days_remaining <= 2:
            insights.append("Batch approaching expiry - immediate action required")
        elif batch.days_remaining <= 5:
            insights.append("Batch nearing expiry - plan for sale or processing")
        
        return insights
    
    def _generate_report_recommendations(self, stats, insights, batch):
        """Generate recommendations from report data"""
        recommendations = []
        
        # Based on statistics
        if stats['temperature_stats']['stability'] == 'Unstable':
            recommendations.append("Stabilize temperature control system")
        
        if stats['humidity_stats']['stability'] == 'Unstable':
            recommendations.append("Improve humidity control for consistency")
        
        # Based on insights
        for insight in insights:
            if 'too low' in insight.lower() or 'too high' in insight.lower():
                recommendations.append(f"Adjust {insight.split()[0].lower()} to optimal range")
            elif 'deteriorating' in insight.lower():
                recommendations.append("Implement accelerated sales strategy")
            elif 'approaching expiry' in insight.lower():
                recommendations.append("Schedule immediate processing or discount sale")
        
        # General recommendations
        if 'Rotten' in stats['quality_distribution'] and stats['quality_distribution']['Rotten'] > 0:
            recommendations.append("Remove rotten fruits to prevent contamination")
        
        if stats['average_confidence'] < 0.7:
            recommendations.append("Consider retraining AI model with updated data")
        
        # Storage optimization
        recommendations.append("Regularly monitor and log storage conditions")
        recommendations.append("Implement first-in-first-out (FIFO) inventory management")
        
        return recommendations
    
    def predict_sales_demand(self, fruit_type, historical_data, market_factors=None):
        """Predict sales demand for a fruit type"""
        try:
            # Convert historical data to DataFrame
            df = pd.DataFrame(historical_data)
            
            if len(df) < 10:
                return {'error': 'Insufficient historical data for prediction'}
            
            # Ensure required columns
            required_cols = ['date', 'quantity_sold']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {'error': f'Missing required columns: {missing_cols}'}
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Add time features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Simple moving average prediction
            window = min(7, len(df))
            last_values = df['quantity_sold'].tail(window).values
            
            # Predict next 7 days
            predictions = []
            for i in range(7):
                predicted = np.mean(last_values[-window:]) if len(last_values) >= window else np.mean(last_values)
                predictions.append(predicted)
                last_values = np.append(last_values, predicted)
            
            # Adjust for market factors
            if market_factors:
                seasonality = market_factors.get('seasonality', 1.0)
                demand_factor = market_factors.get('demand', 1.0)
                holiday_factor = market_factors.get('holiday', 1.0)
                
                predictions = [p * seasonality * demand_factor * holiday_factor for p in predictions]
            
            # Calculate statistics
            avg_prediction = np.mean(predictions)
            total_prediction = np.sum(predictions)
            
            # Generate recommendations
            recommendations = []
            if avg_prediction > df['quantity_sold'].mean() * 1.5:
                recommendations.append("High demand predicted - increase stock levels")
            elif avg_prediction < df['quantity_sold'].mean() * 0.5:
                recommendations.append("Low demand predicted - reduce stock levels")
            
            # Identify peak days
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peak_day_idx = np.argmax(predictions)
            peak_day = day_names[peak_day_idx]
            
            return {
                'success': True,
                'fruit_type': fruit_type,
                'predictions': {
                    'next_7_days': [float(p) for p in predictions],
                    'average_daily': float(avg_prediction),
                    'total_weekly': float(total_prediction),
                    'peak_day': peak_day,
                    'peak_quantity': float(predictions[peak_day_idx])
                },
                'statistics': {
                    'historical_avg': float(df['quantity_sold'].mean()),
                    'historical_std': float(df['quantity_sold'].std()),
                    'data_points': len(df),
                    'time_period': f"{df['date'].min().date()} to {df['date'].max().date()}"
                },
                'recommendations': recommendations,
                'confidence': min(0.9, len(df) / 100)  # Confidence based on data size
            }
            
        except Exception as e:
            logger.error(f"Error predicting sales demand: {e}")
            return {'error': str(e)}
    
    def create_prescription(self, batch_id, target_outcome='maintain_quality'):
        """Create AI-powered prescription for fruit batch management"""
        try:
            from bika.models import FruitBatch
            
            batch = FruitBatch.objects.get(id=batch_id)
            
            # Analyze current state
            analysis = self.analyze_batch_trends(batch_id, days=3)
            
            if 'error' in analysis:
                return {'error': analysis['error']}
            
            # Generate prescription based on target outcome
            prescription = {
                'batch_info': {
                    'batch_number': batch.batch_number,
                    'fruit_type': batch.fruit_type.name,
                    'current_quality': analysis['statistics']['current_quality'],
                    'days_remaining': batch.days_remaining
                },
                'target_outcome': target_outcome,
                'current_analysis': analysis,
                'prescription_steps': [],
                'expected_results': {},
                'monitoring_plan': []
            }
            
            # Define prescription steps based on target outcome
            if target_outcome == 'maintain_quality':
                prescription['prescription_steps'] = [
                    "Maintain temperature at 4-8°C",
                    "Keep humidity at 85-95%",
                    "Minimize light exposure",
                    "Ensure proper ventilation",
                    "Monitor daily for quality changes"
                ]
                
                prescription['expected_results'] = {
                    'quality_maintenance': 'Stable or improved quality',
                    'shelf_life_extension': 'Up to 20% longer shelf life',
                    'waste_reduction': 'Up to 30% less waste'
                }
            
            elif target_outcome == 'accelerate_ripening':
                prescription['prescription_steps'] = [
                    "Increase temperature to 18-22°C",
                    "Introduce ethylene gas (controlled)",
                    "Store with ripe bananas or apples",
                    "Monitor every 12 hours",
                    "Harvest when desired ripeness achieved"
                ]
                
                prescription['expected_results'] = {
                    'ripening_time': '2-3 days faster ripening',
                    'uniformity': 'More uniform ripening',
                    'quality': 'Maintained fruit quality'
                }
            
            elif target_outcome == 'extend_shelf_life':
                prescription['prescription_steps'] = [
                    "Reduce temperature to 2-4°C",
                    "Maintain humidity at 90-95%",
                    "Use ethylene absorbers",
                    "Implement controlled atmosphere (5% O2, 5% CO2)",
                    "Minimize handling and vibration"
                ]
                
                prescription['expected_results'] = {
                    'shelf_life_extension': 'Up to 50% longer shelf life',
                    'quality_preservation': 'Maintained freshness',
                    'economic_benefit': 'Reduced losses, higher profits'
                }
            
            # Monitoring plan
            prescription['monitoring_plan'] = [
                "Check temperature every 6 hours",
                "Monitor humidity daily",
                "Record quality observations daily",
                "Take photos for visual documentation",
                "Update prescription based on results"
            ]
            
            # Success metrics
            prescription['success_metrics'] = {
                'quality_score': 'Maintain or improve by 10%',
                'shelf_life': 'Extend by target amount',
                'economic_value': 'Maximize ROI',
                'customer_satisfaction': 'Meet quality expectations'
            }
            
            return prescription
            
        except Exception as e:
            logger.error(f"Error creating prescription: {e}")
            return {'error': str(e)}


# ==================== SERVICE FACTORY ====================

class AIServiceFactory:
    """Factory for creating AI service instances"""
    
    @staticmethod
    def create_service(service_type='enhanced'):
        """Create AI service instance based on type"""
        if service_type == 'enhanced':
            return EnhancedBikaAIService()
        elif service_type == 'basic':
            return BikaAIService()
        else:
            raise ValueError(f"Unknown service type: {service_type}")
    
    @staticmethod
    def get_available_services():
        """Get list of available AI services"""
        return {
            'enhanced': 'Enhanced AI Service (recommended)',
            'basic': 'Basic AI Service (lightweight)'
        }


# ==================== GLOBAL INSTANCES ====================

# Create global AI service instances
basic_ai_service = BikaAIService()
enhanced_ai_service = EnhancedBikaAIService()
ai_service_factory = AIServiceFactory()

# Default service (can be configured in settings)
DEFAULT_AI_SERVICE_TYPE = getattr(settings, 'BIKA_AI_SERVICE_TYPE', 'enhanced')

if DEFAULT_AI_SERVICE_TYPE == 'enhanced':
    ai_service = enhanced_ai_service
else:
    ai_service = basic_ai_service

# Export
__all__ = [
    'BikaAIService',
    'EnhancedBikaAIService',
    'AIServiceFactory',
    'basic_ai_service',
    'enhanced_ai_service',
    'ai_service_factory',
    'ai_service'
]