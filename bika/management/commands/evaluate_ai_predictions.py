from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
import logging
import numpy as np
# from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score # Example metrics - sklearn not installed by default

from bika.models import (
    Product, ProductAIInsights, AIPredictionAccuracy, TrainedModel,
    Order, OrderItem, InventoryHistory, FruitQualityReading, InventoryItem
)

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Evaluates the accuracy of past AI predictions (stock, demand, quality) and records metrics.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days-back',
            type=int,
            default=30,
            help='Number of days back to evaluate predictions made within that period.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting AI prediction evaluation..."))
        
        days_back = options['days_back']
        evaluation_period_end = timezone.now().date()
        evaluation_period_start = evaluation_period_end - timedelta(days=days_back)

        insights_to_evaluate = ProductAIInsights.objects.filter(
            last_analyzed__date__range=(evaluation_period_start, evaluation_period_end),
            product__isnull=False
        ).select_related('product', 'analysis_model_used').iterator()

        evaluated_count = 0
        
        for insight in insights_to_evaluate:
            product = insight.product
            model_used = insight.analysis_model_used
            
            # --- Evaluate Stock Level Prediction ---
            if insight.predicted_stock_level is not None:
                try:
                    # Get actual stock level at the end of the prediction period
                    # This is a simplified approach; ideally, stock history needs to be granular
                    # For simplicity, we'll compare against the current stock at product level.
                    # A more accurate comparison would require historical InventoryItem data.
                    actual_stock_at_period_end = product.stock_quantity # Simplified: current stock
                    predicted_stock_level = insight.predicted_stock_level
                    
                    mae = abs(predicted_stock_level - actual_stock_at_period_end)
                    
                    AIPredictionAccuracy.objects.create(
                        model=model_used,
                        prediction_type='stock_level',
                        evaluation_date=timezone.now(),
                        metric_name='MAE',
                        metric_value=float(mae),
                        period_start=insight.last_analyzed.date(),
                        period_end=evaluation_period_end,
                        product=product,
                        notes=f"Predicted: {predicted_stock_level}, Actual: {actual_stock_at_period_end}"
                    )
                    self.stdout.write(f"Evaluated stock prediction for {product.name}. MAE: {mae:.2f}")
                    evaluated_count += 1
                except Exception as e:
                    logger.error(f"Error evaluating stock prediction for {product.name} (ID: {product.id}): {e}", exc_info=True)

            # --- Evaluate Demand Forecast ---
            if insight.demand_forecast_next_7_days is not None:
                try:
                    # Get actual sales for the next 7 days from when the insight was made
                    actual_demand_start = insight.last_analyzed.date()
                    actual_demand_end = actual_demand_start + timedelta(days=7)
                    
                    actual_sales_count = OrderItem.objects.filter(
                        product=product,
                        order__created_at__date__range=(actual_demand_start, actual_demand_end),
                        order__status__in=['confirmed', 'shipped', 'delivered']
                    ).aggregate(total_quantity=Sum('quantity'))['total_quantity'] or 0
                    
                    predicted_demand = insight.demand_forecast_next_7_days
                    mae_demand = abs(predicted_demand - actual_sales_count)
                    
                    AIPredictionAccuracy.objects.create(
                        model=model_used,
                        prediction_type='demand_forecast',
                        evaluation_date=timezone.now(),
                        metric_name='MAE_7_day',
                        metric_value=float(mae_demand),
                        period_start=actual_demand_start,
                        period_end=actual_demand_end,
                        product=product,
                        notes=f"Predicted: {predicted_demand}, Actual: {actual_sales_count}"
                    )
                    self.stdout.write(f"Evaluated demand forecast for {product.name}. MAE: {mae_demand:.2f}")
                    evaluated_count += 1
                except Exception as e:
                    logger.error(f"Error evaluating demand forecast for {product.name} (ID: {product.id}): {e}", exc_info=True)
            
            # --- Evaluate Quality Classification ---
            if insight.predicted_quality_class:
                try:
                    # For quality, it's harder to get "actual_class" directly on Product level.
                    # This would typically involve manual inspection results or post-purchase feedback.
                    # For simplicity, we'll try to find a FruitQualityReading for associated batches/items
                    
                    # Assuming there's a way to link Product to a FruitBatch
                    # or an InventoryItem to a FruitQualityReading
                    # This requires more complex data linking.
                    # For this example, we'll skip direct quality evaluation at product level
                    # unless a direct 'actual_quality_class' field is added to ProductAIInsights.
                    pass 
                except Exception as e:
                    logger.error(f"Error evaluating quality prediction for {product.name} (ID: {product.id}): {e}", exc_info=True)

        self.stdout.write(self.style.SUCCESS(
            f"Finished AI prediction evaluation. Processed {evaluated_count} predictions."
        ))