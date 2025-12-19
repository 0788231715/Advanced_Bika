# fixed_real_world_trainer.py - FIXED VERSION
print("🚀 REAL-WORLD DATASET TRAINER (FIXED)")
print("=" * 60)

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')

class FixedRealWorldTrainer:
    """
    Fixed trainer that properly handles both classification and regression
    """
    
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_columns = []
        self.target_column = None
        self.dataset_info = {}
        self.problem_type = None
        
    def load_dataset(self, file_path):
        """Load dataset with proper error handling"""
        print(f"\n📂 Loading dataset: {os.path.basename(file_path)}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"   ✓ Loaded with {encoding} encoding")
                    break
                except:
                    continue
            else:
                df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            
            print(f"   ✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   ✓ Columns: {list(df.columns)}")
            
            # Show sample
            print(f"\n   📊 First 3 rows:")
            print(df.head(3).to_string())
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error loading: {e}")
            return None
    
    def analyze_dataset(self, df):
        """Analyze dataset with proper target detection"""
        print(f"\n📈 DATASET ANALYSIS")
        print("=" * 60)
        
        analysis = {
            'basic_info': {'rows': df.shape[0], 'columns': df.shape[1]},
            'column_types': {},
            'missing_values': {},
            'unique_counts': {}
        }
        
        # Column analysis
        print(f"\n📊 Column Analysis:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique = df[col].nunique()
            missing = df[col].isnull().sum()
            
            analysis['column_types'][col] = dtype
            analysis['unique_counts'][col] = unique
            analysis['missing_values'][col] = int(missing)
            
            print(f"   {col:<15} {dtype:<10} {unique:>4} unique  {missing:>4} missing")
            
            if missing > 0:
                print(f"      ⚠️  {missing/len(df)*100:.1f}% missing values")
        
        # Suggest target column
        print(f"\n🎯 Target Column Suggestions:")
        suggestions = []
        
        # Look for 'Class' column (common in your dataset)
        if 'Class' in df.columns:
            suggestions.append(('Class', 'Binary classification (Good/Bad)'))
        
        # Look for quality-related columns
        quality_cols = [col for col in df.columns if any(word in col.lower() 
                       for word in ['class', 'quality', 'grade', 'type', 'category'])]
        for col in quality_cols:
            if col != 'Class':
                suggestions.append((col, 'Quality classification'))
        
        # Look for numeric columns that could be regression targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            unique_vals = df[col].nunique()
            if 10 < unique_vals < 100:  # Likely regression
                suggestions.append((col, f'Regression ({unique_vals} unique values)'))
            elif unique_vals <= 10:  # Likely classification
                suggestions.append((col, f'Classification ({unique_vals} classes)'))
        
        for i, (col, reason) in enumerate(suggestions[:5], 1):
            print(f"   {i}. {col:<15} - {reason}")
        
        analysis['suggestions'] = suggestions
        
        self.dataset_info = analysis
        return analysis
    
    def preprocess_for_target(self, df, target_column):
        """
        Preprocess based on target column type
        """
        print(f"\n🛠️  Preprocessing for target: {target_column}")
        print("-" * 40)
        
        # Create a copy
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in [np.float64, np.int64]:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        self.target_column = target_column
        
        # Determine problem type
        if y.dtype in [np.float64, np.int64]:
            unique_vals = y.nunique()
            if unique_vals <= 10:  # Small number of unique numeric values
                self.problem_type = 'classification'
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                print(f"   Classification: {unique_vals} classes")
                print(f"   Classes: {list(self.label_encoder.classes_)}")
                y = y_encoded
            else:
                self.problem_type = 'regression'
                print(f"   Regression: {unique_vals} unique values")
                print(f"   Range: {y.min():.2f} to {y.max():.2f}")
        else:
            self.problem_type = 'classification'
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            print(f"   Classification: {len(self.label_encoder.classes_)} classes")
            print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\n✅ Preprocessing Complete:")
        print(f"   Problem type: {self.problem_type}")
        print(f"   Features: {X.shape[1]} columns")
        print(f"   Samples: {X.shape[0]} rows")
        
        return X_scaled, y
    
    def train_models(self, X, y):
        """Train appropriate models based on problem type"""
        print(f"\n🚀 Training {self.problem_type.upper()} Models")
        print("=" * 60)
        
        # Split data
        if self.problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"   Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Testing samples: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Get model configurations
        if self.problem_type == 'classification':
            model_configs = {
                'random_forest': {
                    'name': 'Random Forest',
                    'class': RandomForestClassifier,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 20,
                        'min_samples_split': 10,
                        'random_state': 42,
                        'n_jobs': -1,
                        'class_weight': 'balanced'
                    }
                },
                'xgboost': {
                    'name': 'XGBoost',
                    'class': XGBClassifier,
                    'params': {
                        'n_estimators': 150,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'random_state': 42,
                        'use_label_encoder': False,
                        'eval_metric': 'mlogloss'
                    }
                },
                'knn': {
                    'name': 'K-Nearest Neighbors',
                    'class': KNeighborsClassifier,
                    'params': {
                        'n_neighbors': 7,
                        'weights': 'distance',
                        'algorithm': 'auto'
                    }
                },
                'gradient_boosting': {
                    'name': 'Gradient Boosting',
                    'class': GradientBoostingClassifier,
                    'params': {
                        'n_estimators': 150,
                        'learning_rate': 0.1,
                        'max_depth': 6,
                        'random_state': 42
                    }
                },
                'svm': {
                    'name': 'Support Vector Machine',
                    'class': SVC,
                    'params': {
                        'C': 1.0,
                        'kernel': 'rbf',
                        'probability': True,
                        'random_state': 42,
                        'class_weight': 'balanced'
                    }
                }
            }
        else:  # Regression
            model_configs = {
                'random_forest': {
                    'name': 'Random Forest',
                    'class': RandomForestRegressor,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 20,
                        'min_samples_split': 10,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'xgboost': {
                    'name': 'XGBoost',
                    'class': XGBRegressor,
                    'params': {
                        'n_estimators': 150,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'random_state': 42
                    }
                },
                'knn': {
                    'name': 'K-Nearest Neighbors',
                    'class': KNeighborsRegressor,
                    'params': {
                        'n_neighbors': 7,
                        'weights': 'distance',
                        'algorithm': 'auto'
                    }
                },
                'gradient_boosting': {
                    'name': 'Gradient Boosting',
                    'class': GradientBoostingRegressor,
                    'params': {
                        'n_estimators': 150,
                        'learning_rate': 0.1,
                        'max_depth': 6,
                        'random_state': 42
                    }
                },
                'svm': {
                    'name': 'Support Vector Machine',
                    'class': SVR,
                    'params': {
                        'C': 1.0,
                        'kernel': 'rbf'
                    }
                }
            }
        
        # Train models
        self.model_metrics = {}
        
        for model_key, config in model_configs.items():
            try:
                print(f"\n📊 Training {config['name']}...")
                
                # Initialize and train
                model = config['class'](**config['params'])
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if self.problem_type == 'classification':
                    metrics = self._calculate_classification_metrics(y_test, y_pred, model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                
                # Cross-validation
                cv_scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
                cv_scores = cross_val_score(model, X, y, cv=5, scoring=cv_scoring)
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                # Store results
                metrics['model_name'] = config['name']
                self.models[model_key] = {
                    'model': model,
                    'metrics': metrics,
                    'config': config['params']
                }
                self.model_metrics[model_key] = metrics
                
                # Print results
                if self.problem_type == 'classification':
                    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
                    print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
                    print(f"   📈 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                else:
                    print(f"   ✅ R² Score: {metrics['r2_score']:.4f}")
                    print(f"   📊 RMSE: {metrics['rmse']:.4f}")
                    print(f"   📈 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:80]}")
                continue
        
        # Select best model
        self._select_best_model()
        
        return self.model_metrics
    
    def _calculate_classification_metrics(self, y_true, y_pred, model, X_test):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC if probabilities available
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_true)) > 1:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                else:
                    metrics['roc_auc'] = 0.0
            except:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Weighted score
        metrics['weighted_score'] = (
            metrics['accuracy'] * 0.4 +
            metrics['f1_score'] * 0.4 +
            metrics.get('roc_auc', 0.5) * 0.2
        )
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
        
        # Use R² as main score for regression
        metrics['weighted_score'] = max(0, metrics['r2_score'])
        
        return metrics
    
    def _select_best_model(self):
        """Select best model based on weighted score"""
        if not self.model_metrics:
            return None
        
        best_score = -float('inf')
        best_key = None
        
        for key, metrics in self.model_metrics.items():
            if metrics['weighted_score'] > best_score:
                best_score = metrics['weighted_score']
                best_key = key
        
        if best_key:
            self.best_model = {
                'key': best_key,
                'name': self.model_metrics[best_key]['model_name'],
                'metrics': self.model_metrics[best_key],
                'model': self.models[best_key]['model']
            }
        
        return self.best_model
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "=" * 80)
        print(f"🏆 {self.problem_type.upper()} TRAINING RESULTS")
        print("=" * 80)
        
        # Sort models by performance
        sorted_models = sorted(
            self.model_metrics.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )
        
        if self.problem_type == 'classification':
            print(f"\n📋 MODEL RANKING")
            print("-" * 80)
            print(f"{'Rank':<5} {'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Score':<10}")
            print("-" * 80)
            
            for rank, (key, metrics) in enumerate(sorted_models, 1):
                print(f"{rank:<5} {metrics['model_name']:<25} "
                      f"{metrics['accuracy']:.4f}     "
                      f"{metrics['f1_score']:.4f}     "
                      f"{metrics['precision']:.4f}     "
                      f"{metrics['recall']:.4f}     "
                      f"{metrics['weighted_score']:.4f}")
        else:
            print(f"\n📋 MODEL RANKING")
            print("-" * 80)
            print(f"{'Rank':<5} {'Model':<25} {'R² Score':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'Score':<10}")
            print("-" * 80)
            
            for rank, (key, metrics) in enumerate(sorted_models, 1):
                print(f"{rank:<5} {metrics['model_name']:<25} "
                      f"{metrics['r2_score']:.4f}     "
                      f"{metrics['rmse']:.4f}     "
                      f"{metrics['mae']:.4f}     "
                      f"{metrics['mape']:.1f}%    "
                      f"{metrics['weighted_score']:.4f}")
        
        print("-" * 80)
        
        # Best model details
        if self.best_model:
            print(f"\n🎯 BEST MODEL: {self.best_model['name']}")
            print("-" * 80)
            
            best = self.best_model['metrics']
            if self.problem_type == 'classification':
                print(f"   Accuracy:    {best['accuracy']:.4f}")
                print(f"   F1-Score:    {best['f1_score']:.4f}")
                print(f"   Precision:   {best['precision']:.4f}")
                print(f"   Recall:      {best['recall']:.4f}")
                if 'roc_auc' in best:
                    print(f"   ROC-AUC:     {best['roc_auc']:.4f}")
                print(f"   CV Score:    {best['cv_mean']:.4f} (±{best['cv_std']:.4f})")
            else:
                print(f"   R² Score:    {best['r2_score']:.4f}")
                print(f"   RMSE:        {best['rmse']:.4f}")
                print(f"   MAE:         {best['mae']:.4f}")
                print(f"   MAPE:        {best['mape']:.1f}%")
                print(f"   CV Score:    {best['cv_mean']:.4f} (±{best['cv_std']:.4f})")
        
        # Performance interpretation
        print(f"\n💡 PERFORMANCE INTERPRETATION:")
        print("-" * 80)
        
        if self.problem_type == 'classification':
            best_acc = self.best_model['metrics']['accuracy'] if self.best_model else 0
            
            if best_acc > 0.90:
                print("   🏆 EXCELLENT: >90% accuracy - Model works exceptionally well")
            elif best_acc > 0.80:
                print("   👍 VERY GOOD: 80-90% accuracy - Ready for production")
            elif best_acc > 0.70:
                print("   ✅ GOOD: 70-80% accuracy - Suitable for most applications")
            elif best_acc > 0.60:
                print("   ⚠️  FAIR: 60-70% accuracy - May need improvement")
            else:
                print("   🚨 NEEDS WORK: <60% accuracy - Consider different approach")
        else:
            best_r2 = self.best_model['metrics']['r2_score'] if self.best_model else 0
            
            if best_r2 > 0.9:
                print("   🏆 EXCELLENT: >0.9 R² - Model explains most variance")
            elif best_r2 > 0.7:
                print("   👍 VERY GOOD: 0.7-0.9 R² - Strong predictive power")
            elif best_r2 > 0.5:
                print("   ✅ GOOD: 0.5-0.7 R² - Reasonable predictions")
            elif best_r2 > 0.3:
                print("   ⚠️  FAIR: 0.3-0.5 R² - Limited predictive power")
            else:
                print("   🚨 NEEDS WORK: <0.3 R² - Consider different features/approach")
    
    def save_results(self, original_df):
        """Save all results properly"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model
        if self.best_model:
            model_filename = f"best_model_{timestamp}.pkl"
            metadata = {
                'model_type': self.best_model['key'],
                'model_name': self.best_model['name'],
                'problem_type': self.problem_type,
                'target_column': self.target_column,
                'feature_columns': self.feature_columns,
                'metrics': self.best_model['metrics'],
                'training_date': timestamp,
                'dataset_shape': original_df.shape
            }
            
            if self.problem_type == 'classification' and self.label_encoder:
                metadata['label_classes'] = self.label_encoder.classes_.tolist()
            
            model_data = {
                'model': self.best_model['model'],
                'scaler': self.scaler,
                'metadata': metadata
            }
            
            if self.problem_type == 'classification' and self.label_encoder:
                model_data['label_encoder'] = self.label_encoder
            
            joblib.dump(model_data, model_filename)
            print(f"\n💾 Best model saved: {model_filename}")
        
        # Save training report
        report_data = {
            'training_summary': {
                'best_model': self.best_model['name'] if self.best_model else None,
                'problem_type': self.problem_type,
                'target_column': self.target_column,
                'models_trained': len(self.models),
                'dataset_info': self.dataset_info
            },
            'model_comparison': self.model_metrics,
            'best_model_details': self.best_model
        }
        
        report_filename = f"training_report_{timestamp}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved: {report_filename}")
        
        # Save predictions if classification
        if self.best_model and self.problem_type == 'classification':
            X, y = self.preprocess_for_target(original_df, self.target_column)
            predictions = self.best_model['model'].predict(X)
            
            if self.label_encoder:
                original_df['predicted'] = self.label_encoder.inverse_transform(predictions)
            else:
                original_df['predicted'] = predictions
            
            predictions_filename = f"predictions_{timestamp}.csv"
            original_df.to_csv(predictions_filename, index=False)
            print(f"📊 Predictions saved: {predictions_filename}")


def main():
    """Main function with proper target selection"""
    print("\n" + "=" * 60)
    print("🎯 REAL DATASET TRAINER")
    print("=" * 60)
    
    # Find datasets
    folder = input("Enter dataset folder (or press Enter for current): ").strip()
    if not folder:
        folder = "."
    
    # Look for CSV files
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    print(f"\n📁 Found {len(csv_files)} CSV file(s):")
    for i, file in enumerate(csv_files, 1):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   {i}. {os.path.basename(file)} ({size_mb:.2f} MB)")
    
    # Select file
    try:
        choice = int(input(f"\nSelect file (1-{len(csv_files)}): "))
        file_path = csv_files[choice - 1]
    except:
        print("❌ Invalid selection!")
        return
    
    # Load dataset
    trainer = FixedRealWorldTrainer()
    df = trainer.load_dataset(file_path)
    
    if df is None:
        return
    
    # Analyze dataset
    trainer.analyze_dataset(df)
    
    # Show column suggestions
    print(f"\n📊 Dataset columns: {list(df.columns)}")
    
    # Ask for target column
    target = input("\n🎯 Enter target column name: ").strip()
    
    if target not in df.columns:
        print(f"❌ Column '{target}' not found in dataset!")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    print(f"\n✅ Target selected: {target}")
    print(f"   Type: {df[target].dtype}")
    print(f"   Unique values: {df[target].nunique()}")
    print(f"   Sample values: {df[target].unique()[:5]}")
    
    # Confirm problem type
    if df[target].dtype in [np.float64, np.int64]:
        unique_vals = df[target].nunique()
        if unique_vals <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
    else:
        problem_type = 'classification'
    
    print(f"\n🔍 Problem type detected: {problem_type}")
    
    # Start training
    print(f"\n🚀 Starting training...")
    print("=" * 60)
    
    # Preprocess and train
    X, y = trainer.preprocess_for_target(df, target)
    results = trainer.train_models(X, y)
    
    # Display results
    trainer.display_results()
    
    # Save results
    trainer.save_results(df)
    
    print(f"\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
print("\n🎯 Training script finished!")