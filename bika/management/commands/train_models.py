
import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from bika.ai_models import FruitQualityPredictor

class Command(BaseCommand):
    help = 'Train and evaluate multiple AI models for fruit quality prediction'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting model training and evaluation...'))

        # 1. Define models to train
        model_types = [
            'random_forest',
            'gradient_boosting',
            'svm',
            'knn',
            'xgboost'
        ]

        # 2. Load and prepare data
        dataset_path = os.path.join(settings.MEDIA_ROOT, 'fruit_datasets', 'sample_fruit_dataset.csv')
        if not os.path.exists(dataset_path):
            self.stdout.write(self.style.ERROR(f'Dataset not found at: {dataset_path}'))
            return

        df = pd.read_csv(dataset_path)

        # Rename columns to match what FruitQualityPredictor expects
        column_mapping = {
            'Fruit': 'fruit_type',
            'Temp': 'temperature',
            'Humid (%)': 'humidity',
            'Light (Fux)': 'light_intensity',
            'CO2 (pmm)': 'co2_level',
            'Class': 'quality_class'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Create a temporary path for the cleaned data
        temp_dataset_dir = os.path.join(settings.MEDIA_ROOT, 'temp_datasets')
        os.makedirs(temp_dataset_dir, exist_ok=True)
        temp_dataset_path = os.path.join(temp_dataset_dir, 'cleaned_fruit_dataset.csv')
        df.to_csv(temp_dataset_path, index=False)


        all_metrics = []

        for model_type in model_types:
            self.stdout.write(self.style.HTTP_INFO(f'--- Training {model_type} model ---'))
            
            # 3. Train model
            predictor = FruitQualityPredictor(model_type=model_type)
            
            X, y, _ = predictor.load_fruit_dataset(temp_dataset_path)

            if X is None or y is None:
                self.stdout.write(self.style.ERROR(f'Failed to load data for {model_type}. Skipping.'))
                continue

            metrics = predictor.train_model(X, y, use_grid_search=False)

            if 'error' in metrics:
                self.stdout.write(self.style.ERROR(f'Error training {model_type}: {metrics["error"]}'))
                continue

            self.stdout.write(self.style.SUCCESS(f'Successfully trained {model_type} model.'))
            
            # Save the model
            timestamp = int(pd.Timestamp.now().timestamp())
            model_filename = f'fruit_quality_model_{predictor.model_type}_{timestamp}.pkl'
            model_dir = os.path.join(settings.MEDIA_ROOT, 'fruit_models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_filename)
            predictor.save_model(model_path)
            
            metrics['model_name'] = model_type
            all_metrics.append(metrics)

        # 4. Display results
        self.stdout.write(self.style.SUCCESS('\n--- Model Performance Summary ---'))
        
        if not all_metrics:
            self.stdout.write(self.style.WARNING('No models were trained successfully.'))
            return

        # Find the best model
        best_model = max(all_metrics, key=lambda x: x['accuracy'])

        header = f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}"
        self.stdout.write(header)
        self.stdout.write('-' * len(header))

        for metrics in all_metrics:
            is_best = ' (Best)' if metrics['model_name'] == best_model['model_name'] else ''
            self.stdout.write(
                f"{metrics['model_name']:<20} | "
                f"{metrics['accuracy']:.4f}{'':<4} | "
                f"{metrics['precision']:.4f}{'':<4} | "
                f"{metrics['recall']:.4f}{'':<4} | "
                f"{metrics['f1_score']:.4f}{'':<4}{is_best}"
            )
        
        self.stdout.write(self.style.SUCCESS(f"\nBest performing model is: {best_model['model_name']}"))

        # Clean up temporary file
        os.remove(temp_dataset_path)
        self.stdout.write(self.style.SUCCESS('Model training and evaluation complete.'))
