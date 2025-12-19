# bika/management/commands/scan_products.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from bika.ai_integration.services import FruitAIService
from bika.models import FruitProduct

class Command(BaseCommand):
    help = 'Scan all products and generate AI predictions and alerts'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--product-id',
            type=int,
            help='Scan specific product only'
        )
        parser.add_argument(
            '--batch',
            action='store_true',
            help='Scan products in batches'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting product quality scan...'))
        
        # Initialize AI service
        ai_service = FruitAIService()
        
        # Get products to scan
        if options['product_id']:
            products = FruitProduct.objects.filter(id=options['product_id'])
        else:
            # Scan active products
            products = FruitProduct.objects.filter(
                status='active',
                expiry_date__gt=timezone.now().date()
            )
        
        total_products = products.count()
        self.stdout.write(f"📊 Scanning {total_products} products...")
        
        results = {
            'scanned': 0,
            'predictions': 0,
            'alerts': 0,
            'errors': 0
        }
        
        # Scan products
        for i, product in enumerate(products, 1):
            try:
                self.stdout.write(f"\n[{i}/{total_products}] Scanning: {product.name}...")
                
                # Make predictions
                prediction_result = ai_service.predict_product_quality(product.id)
                
                if 'error' in prediction_result:
                    self.stdout.write(self.style.ERROR(f"   ❌ Error: {prediction_result['error']}"))
                    results['errors'] += 1
                else:
                    predictions = len(prediction_result.get('predictions', []))
                    alerts = len(prediction_result.get('alerts', []))
                    
                    results['predictions'] += predictions
                    results['alerts'] += alerts
                    
                    self.stdout.write(f"   ✅ Predictions: {predictions}")
                    self.stdout.write(f"   ⚠️  Alerts: {alerts}")
                
                results['scanned'] += 1
                
                # Small delay for batch processing
                if options['batch'] and i % 10 == 0:
                    self.stdout.write(f"   ⏸️  Batch checkpoint: {i}/{total_products}")
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"   ❌ Failed: {str(e)}"))
                results['errors'] += 1
        
        # Summary
        self.stdout.write(self.style.SUCCESS("\n" + "="*50))
        self.stdout.write(self.style.SUCCESS("📊 SCAN COMPLETED"))
        self.stdout.write(self.style.SUCCESS("="*50))
        self.stdout.write(f"Products scanned: {results['scanned']}")
        self.stdout.write(f"Predictions made: {results['predictions']}")
        self.stdout.write(f"Alerts generated: {results['alerts']}")
        self.stdout.write(f"Errors: {results['errors']}")
        
        if results['alerts'] > 0:
            self.stdout.write(self.style.WARNING(f"\n⚠️  {results['alerts']} alerts require attention!"))
        
        self.stdout.write(self.style.SUCCESS("\n✅ Scan completed successfully!"))