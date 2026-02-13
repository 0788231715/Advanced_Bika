from django.core.management.base import BaseCommand
from django.utils import timezone
from decimal import Decimal
import logging

from bika.models import Product, ProductAIInsights, ProductPriceHistory, CustomUser

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Automatically adjusts product pricing based on AI demand forecasts and quality predictions.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting automatic pricing automation..."))
        
        # Get a system user to record price changes
        system_user = CustomUser.objects.filter(is_superuser=True).first()
        if not system_user:
            self.stdout.write(self.style.ERROR("No superuser found to assign as 'changed_by'. Automated pricing will not proceed."))
            return

        products_for_pricing = Product.objects.filter(
            is_active=True,
            product_type='public' # Only adjust pricing for public e-commerce products
        ).select_related('ai_insights_for_product') # Updated related_name

        adjusted_products_count = 0
        
        for product in products_for_pricing:
            # Ensure the product has AI insights
            try:
                ai_insight = product.ai_insights_for_product.latest('last_analyzed') # Get latest insight
            except ProductAIInsights.DoesNotExist:
                logger.info(f"Product {product.name} (ID: {product.id}) has no AI insights. Skipping pricing adjustment.")
                continue

            current_price = product.price
            new_price = current_price
            reason = "Automated AI adjustment: "
            
            # --- Pricing Rules ---
            # Rule 1: High Demand & Good Quality -> Increase Price
            if ai_insight.demand_forecast_next_7_days >= 100 and ai_insight.predicted_quality_class == 'Excellent':
                new_price = current_price * Decimal('1.05') # 5% increase
                reason += "High demand and excellent quality."
            
            # Rule 2: Low Demand & Poor Quality/Near Expiry -> Decrease Price
            elif ai_insight.demand_forecast_next_7_days <= 10 and (
                ai_insight.predicted_quality_class in ['Poor', 'Critical'] or 
                (ai_insight.predicted_expiry_date and ai_insight.predicted_expiry_date < timezone.now().date() + timezone.timedelta(days=7))
            ):
                new_price = current_price * Decimal('0.90') # 10% decrease
                reason += "Low demand and poor quality/near expiry."
            
            # Rule 3: Moderate Demand & Fair Quality -> Small Adjustment or Maintain
            elif ai_insight.demand_forecast_next_7_days >= 50 and ai_insight.predicted_quality_class == 'Good':
                # Example: slight increase if price is below a target
                target_price = product.cost_price * Decimal('1.5') # Example target margin
                if new_price < target_price:
                    new_price = min(current_price * Decimal('1.02'), target_price) # Max 2% increase, up to target
                    reason += "Moderate demand and good quality, adjusted towards target margin."
                else:
                    reason += "Moderate demand and good quality, price maintained."
            else:
                reason += "No specific rule triggered, price maintained."

            # Ensure price doesn't go below cost price (if available) or above a max limit
            if product.cost_price and new_price < product.cost_price:
                new_price = product.cost_price * Decimal('1.10') # At least 10% above cost
                reason += " Adjusted to be above cost price."
            
            # Add a maximum price cap if necessary (e.g., original price * 2)
            # max_price_cap = product.original_price * Decimal('2.0') 
            # if new_price > max_price_cap:
            #     new_price = max_price_cap
            #     reason += " Capped by maximum price limit."

            # Round to 2 decimal places
            new_price = new_price.quantize(Decimal('0.01'))

            if new_price != current_price:
                # Update product price
                ProductPriceHistory.objects.create(
                    product=product,
                    old_price=current_price,
                    new_price=new_price,
                    changed_by=system_user,
                    reason=reason
                )
                product.price = new_price
                product.save(update_fields=['price'])
                adjusted_products_count += 1
                self.stdout.write(self.style.SUCCESS(
                    f"Adjusted price for {product.name} (ID: {product.id}) from {current_price} to {new_price}. Reason: {reason}"
                ))
            else:
                logger.info(f"Price for {product.name} (ID: {product.id}) remained {current_price}. Reason: {reason}")
        
        self.stdout.write(self.style.SUCCESS(
            f"Finished automatic pricing automation. Adjusted {adjusted_products_count} product prices."
        ))