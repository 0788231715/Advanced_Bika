from django.core.management.base import BaseCommand
from django.db.models import F, Sum, ExpressionWrapper, fields
from django.utils import timezone
import logging

from bika.models import Product, ProductAIInsights, StorageLocation, InventoryTransfer, InventoryItem, CustomUser

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Automates inventory transfers between locations based on demand forecasts and stock levels.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting automatic inventory transfer automation..."))
        
        # Define a threshold for high demand and low stock
        HIGH_DEMAND_THRESHOLD = 50 # Example: if AI predicts demand for >50 units in 7 days
        LOW_STOCK_PERCENTAGE = 0.20 # Example: if current stock is less than 20% of max capacity
        REORDER_QUANTITY_MULTIPLIER = 1.5 # How much to transfer (e.g., 1.5x the difference)

        # Find products with high demand forecasts
        # This is a simplified approach. A more robust solution would link demand to specific locations.
        high_demand_products = ProductAIInsights.objects.filter(
            demand_forecast_next_7_days__gte=HIGH_DEMAND_THRESHOLD,
            product__isnull=False
        ).select_related('product').values_list('product__id', flat=True).distinct()

        processed_transfers_count = 0
        
        for product_id in high_demand_products:
            product = Product.objects.get(id=product_id)
            
            # Aggregate current stock quantity of this product across all locations
            # Note: This is simplified. Ideally, InventoryItem tracks product quantity per location.
            # Assuming Product.stock_quantity is aggregate or we need to iterate InventoryItem
            
            # To simulate location-specific stock:
            # We'll look for InventoryItems of this product
            
            # Find potential source locations (overstocked)
            source_locations = StorageLocation.objects.annotate(
                current_product_stock=Sum('inventory_items__quantity', filter=Q(inventory_items__product=product))
            ).filter(
                current_product_stock__gt=product.low_stock_threshold * REORDER_QUANTITY_MULTIPLIER, # Enough to transfer
                is_active=True
            ).exclude(
                current_product_stock__isnull=True # Exclude locations where product is not found
            ).order_by('-current_product_stock')

            # Find potential destination locations (understocked/high demand)
            destination_locations = StorageLocation.objects.annotate(
                current_product_stock=Sum('inventory_items__quantity', filter=Q(inventory_items__product=product))
            ).filter(
                current_product_stock__lt=product.low_stock_threshold, # Needs stock
                is_active=True
            ).exclude(
                current_product_stock__isnull=False # Exclude locations where product is not found
            ).order_by('current_product_stock')
            
            # If no explicit stock for product in location, assume 0 for understocked.
            # Or consider locations that don't have this product at all for new stocking.

            if not source_locations.exists():
                self.stdout.write(f"No suitable source locations for {product.name}.")
                continue
            
            if not destination_locations.exists():
                self.stdout.write(f"No suitable destination locations for {product.name}.")
                continue

            # Try to create transfers
            for dest_loc in destination_locations:
                for src_loc in source_locations:
                    if src_loc == dest_loc:
                        continue # Cannot transfer to same location
                    
                    # Calculate quantity to transfer
                    # Aim to bring dest_loc stock up to at least low_stock_threshold
                    current_dest_stock = dest_loc.current_product_stock if dest_loc.current_product_stock is not None else 0
                    quantity_needed_at_dest = (product.low_stock_threshold * REORDER_QUANTITY_MULTIPLIER) - current_dest_stock
                    
                    if quantity_needed_at_dest <= 0:
                        continue # Destination doesn't need stock or has enough

                    # Ensure source has enough to transfer
                    quantity_available_at_src = src_loc.current_product_stock
                    transfer_quantity = min(quantity_needed_at_dest, quantity_available_at_src)
                    
                    if transfer_quantity > 0:
                        # Check if a pending transfer already exists for this product, source, and destination
                        existing_transfer = InventoryTransfer.objects.filter(
                            product=product,
                            source_location=src_loc,
                            destination_location=dest_loc,
                            status='pending'
                        ).first()

                        if existing_transfer:
                            self.stdout.write(f"Pending transfer already exists for {product.name} from {src_loc.name} to {dest_loc.name}. Skipping.")
                            continue

                        # Create the transfer request
                        try:
                            # Assuming a default system user for automated transfers
                            system_user = CustomUser.objects.filter(is_superuser=True).first() 
                            if not system_user:
                                self.stdout.write(self.style.ERROR("No superuser found to assign as requested_by. Please create one."))
                                continue

                            InventoryTransfer.objects.create(
                                product=product,
                                quantity=transfer_quantity,
                                source_location=src_loc,
                                destination_location=dest_loc,
                                status='pending',
                                requested_by=system_user,
                                notes=f"Automated transfer based on AI demand forecast ({product.ai_insights_for_product.first().demand_forecast_next_7_days} units in 7 days) and low stock at {dest_loc.name}."
                            )
                            processed_transfers_count += 1
                            self.stdout.write(self.style.SUCCESS(
                                f"Generated transfer for {transfer_quantity}x {product.name} from {src_loc.name} to {dest_loc.name}."
                            ))
                        except Exception as e:
                            logger.error(f"Error creating transfer for {product.name}: {e}", exc_info=True)
                            self.stdout.write(self.style.ERROR(
                                f"Failed to create transfer for {product.name}: {e}"
                            ))

        self.stdout.write(self.style.SUCCESS(
            f"Finished inventory transfer automation. Generated {processed_transfers_count} transfer requests."
        ))