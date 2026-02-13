from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import logging
import pandas as pd
from io import StringIO

from bika.models import Product, ProductAIInsights, AIPredictionAccuracy, TrainedModel, ProductDataset
from bika.services.ai_service import enhanced_ai_service # Import your AI service

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Automatically monitors AI model accuracy and triggers retraining if performance degrades.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.15, # Example MAE threshold for triggering retraining
            help='Accuracy metric threshold for triggering retraining (e.g., MAE).',
        )
        parser.add_argument(
            '--min-accuracy-records',
            type=int,
            default=5,
            help='Minimum number of accuracy records to consider for evaluation.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting automated AI model retraining system..."))
        
        mae_threshold = options['threshold']
        min_records = options['min_accuracy_records']
        
        # Iterate through active AI models and evaluate their recent performance
        active_models = TrainedModel.objects.filter(is_active=True)
        
        for model in active_models:
            self.stdout.write(f"Evaluating model: {model.name} (Type: {model.get_model_type_display()})")
            
            # Retrieve recent accuracy records for this model and its primary prediction type
            # For simplicity, let's assume 'stock_level' predictions are critical for retraining
            recent_accuracy_records = AIPredictionAccuracy.objects.filter(
                model=model,
                prediction_type=model.model_type, # Use model's type as primary prediction type
                metric_name='MAE' # Assuming MAE is a key metric
            ).order_by('-evaluation_date')[:min_records]
            
            if len(recent_accuracy_records) < min_records:
                self.stdout.write(f"Not enough recent accuracy records for {model.name}. Skipping evaluation.")
                continue
            
            # Calculate average MAE for recent predictions
            average_mae = sum(rec.metric_value for rec in recent_accuracy_records) / len(recent_accuracy_records)
            
            self.stdout.write(f"  Recent Average MAE for {model.name}: {average_mae:.4f} (Threshold: {mae_threshold})")

            if average_mae > mae_threshold:
                self.stdout.write(self.style.WARNING(
                    f"  Model {model.name} (Type: {model.get_model_type_display()}) performance has degraded. Triggering retraining..."
                ))
                
                # --- Trigger Retraining ---
                try:
                    # This part needs to be refined based on how your enhanced_ai_service.train_five_models
                    # expects its data. Assuming it can take a ProductDataset or generate its own.
                    # For this example, we'll try to generate a sample dataset.
                    
                    # You would normally fetch the *latest* relevant dataset or dynamically generate it.
                    # This is a placeholder for actual data fetching and preparation.
                    
                    # Example: Get the dataset linked to this model
                    retraining_dataset = model.dataset
                    if not retraining_dataset or not retraining_dataset.data_file:
                        self.stdout.write(self.style.ERROR(f"  No valid dataset found for retraining model {model.name}. Skipping."))
                        continue
                    
                    # Simulate reading data from the dataset file
                    # In a real scenario, this might involve more complex data pipeline
                    # For demonstration, we'll use a dummy CSV from the file system.
                    
                    # Assuming retraining_dataset.data_file points to a CSV
                    # You would need to read this file and convert to a format
                    # expected by enhanced_ai_service.train_five_models
                    
                    # Placeholder for actual data retrieval and target column identification
                    # This is highly dependent on your AI service's implementation
                    
                    # For now, let's assume enhanced_ai_service has a method to retrain a specific model type
                    # using its last used dataset or a default/generated one.
                    
                    # If model.model_type is 'stock_prediction', target_column could be 'stock_quantity'
                    # If 'fruit_quality', target_column could be 'quality_class'
                    target_column_map = {
                        'stock_prediction': 'stock_quantity',
                        'fruit_quality': 'predicted_quality_class',
                        'anomaly_detection': 'is_anomaly', # Example
                        'sales_forecast': 'sales_volume' # Example
                    }
                    target_column = target_column_map.get(model.model_type)

                    if not target_column:
                        self.stdout.write(self.style.ERROR(f"  Cannot determine target column for model type {model.model_type}. Skipping retraining."))
                        continue

                    self.stdout.write(f"  Attempting to retrain model {model.name} with dataset {retraining_dataset.name} and target '{target_column}'...")
                    
                    # This is where the actual AI service call would happen
                    # train_five_models expects a file, so we'll simulate.
                    # In a real system, you'd feed actual preprocessed data.
                    # As a workaround for direct file, we will re-scan all products which would trigger new insights if needed
                    # Or a specific train_model_from_dataset method in enhanced_ai_service
                    
                    # Let's assume enhanced_ai_service has a direct retraining method by model_id and dataset
                    retrain_result = enhanced_ai_service.retrain_model(
                        model_id=model.id, 
                        dataset_file=retraining_dataset.data_file, # Pass the file field
                        target_column=target_column
                    )
                    
                    if retrain_result.get('success'):
                        new_model_id = retrain_result.get('new_model_id')
                        new_accuracy = retrain_result.get('accuracy')
                        
                        if new_model_id and new_accuracy:
                            # Activate the new, better performing model
                            enhanced_ai_service.activate_model(new_model_id)
                            self.stdout.write(self.style.SUCCESS(
                                f"  Successfully retrained and activated new model (ID: {new_model_id}) with accuracy {new_accuracy:.4f}."
                            ))
                        else:
                            self.stdout.write(self.style.WARNING("  Retraining completed, but no new model activated or accuracy improved significantly."))
                    else:
                        self.stdout.write(self.style.ERROR(
                            f"  Retraining failed for model {model.name}: {retrain_result.get('error', 'Unknown error')}."
                        ))
                except Exception as e:
                    logger.error(f"  Unhandled error during retraining for model {model.name}: {e}", exc_info=True)
                    self.stdout.write(self.style.ERROR(f"  Unhandled error during retraining for model {model.name}: {e}"))
            else:
                self.stdout.write(f"  Model {model.name} performance is within acceptable limits. No retraining needed.")

        self.stdout.write(self.style.SUCCESS("Finished automated AI model retraining system."))