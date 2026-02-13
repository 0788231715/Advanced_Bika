from django.core.management.base import BaseCommand
from django.utils import timezone
from decimal import Decimal
import logging
from django.db.models import F # Import F object

from bika.models import Product, ProductAIInsights, PurchaseOrder, PurchaseOrderItem, Supplier

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Automatically generates purchase orders for products with low stock based on AI predictions.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting automatic purchase order generation..."))
        
        # Products with low stock or predicted out-of-stock
        # This is a simplified logic. A real system would combine current stock, AI predictions, and lead times.
        # For this implementation, we'll check products where predicted_stock_level is below reorder point
        # or predicted_out_of_stock_date is in the near future.

        # Find products needing reorder based on AI insights
        products_to_reorder_ids = ProductAIInsights.objects.filter(
            predicted_stock_level__lte=F('product__low_stock_threshold') # Assuming Product has low_stock_threshold for reorder point
        ).values_list('product__id', flat=True)

        products_needing_po = Product.objects.filter(
            id__in=products_to_reorder_ids,
            track_inventory=True,
            default_supplier__isnull=False,
            is_active=True # Only reorder active products
        ).select_related('default_supplier').prefetch_related('ai_insights_for_product').iterator()
        
        generated_pos_count = 0
        
        for product in products_needing_po:
            # Re-check actual stock vs low_stock_threshold or AI predicted stock
            # Using current stock and ProductAIInsights predicted_stock_level
            ai_insight = product.ai_insights_for_product.first() # Assuming one insight per product for now

            if not ai_insight:
                logger.warning(f"Product {product.name} (ID: {product.id}) has no AI insights. Skipping PO generation.")
                continue

            # Determine reorder quantity. This could be complex, involving:
            # - (Reorder Point - current_stock) + safety_stock
            # - AI predicted demand over lead time
            # For simplicity, let's aim to bring stock up to twice the low_stock_threshold,
            # or a fixed quantity if reorder_point is not set.
            current_stock = product.stock_quantity
            reorder_point = product.low_stock_threshold # Using low_stock_threshold as reorder point for simplicity
            
            if current_stock <= reorder_point:
                # Calculate quantity needed to reach a desired stock level (e.g., 2x reorder point or fixed amount)
                quantity_to_order = (reorder_point * 2) - current_stock
                if quantity_to_order <= 0: # Ensure we order at least 1 if stock is critically low
                    quantity_to_order = product.low_stock_threshold # Reorder at least threshold amount

                if quantity_to_order > 0 and product.default_supplier:
                    try:
                        # Check if an open PO already exists for this product and supplier
                        open_po = PurchaseOrder.objects.filter(
                            supplier=product.default_supplier,
                            status__in=['draft', 'pending', 'ordered'],
                            items__product=product
                        ).first()

                        if open_po:
                            # Update existing PO item
                            po_item, created = PurchaseOrderItem.objects.get_or_create(
                                purchase_order=open_po,
                                product=product,
                                defaults={
                                    'quantity': quantity_to_order,
                                    'unit_price': product.cost_price if product.cost_price else product.price
                                }
                            )
                            if not created:
                                # Only increase if AI suggests more, otherwise leave as is
                                if quantity_to_order > po_item.quantity:
                                    po_item.quantity = quantity_to_order
                                    po_item.unit_price = product.cost_price if product.cost_price else product.price
                                    po_item.save()
                                self.stdout.write(f"Updated PO {open_po.order_number} for {product.name} (Qty: {po_item.quantity})")
                            else:
                                self.stdout.write(f"Added {product.name} to existing PO {open_po.order_number} (Qty: {quantity_to_order})")
                        else:
                            # Create new Purchase Order
                            new_po = PurchaseOrder.objects.create(
                                supplier=product.default_supplier,
                                ordered_by=None, # System generated, no specific user
                                expected_delivery_date=timezone.now().date() + timezone.timedelta(days=14), # 2 weeks lead time
                                status='draft' # Start as draft, requires approval
                            )
                            PurchaseOrderItem.objects.create(
                                purchase_order=new_po,
                                product=product,
                                quantity=quantity_to_order,
                                unit_price=product.cost_price if product.cost_price else product.price
                            )
                            generated_pos_count += 1
                            self.stdout.write(self.style.SUCCESS(
                                f"Generated new Purchase Order {new_po.order_number} for {product.name} (Qty: {quantity_to_order})"
                            ))
                    except Exception as e:
                        logger.error(f"Error generating PO for product {product.name} (ID: {product.id}): {e}", exc_info=True)
                        self.stdout.write(self.style.ERROR(
                            f"Failed to generate PO for {product.name}: {e}"
                        ))
                else:
                    self.stdout.write(f"Skipping {product.name}: No default supplier or quantity to order.")
            else:
                self.stdout.write(f"Product {product.name} (ID: {product.id}) does not need reordering yet.")

        self.stdout.write(self.style.SUCCESS(
            f"Finished. Generated/Updated {generated_pos_count} purchase orders."
        ))
