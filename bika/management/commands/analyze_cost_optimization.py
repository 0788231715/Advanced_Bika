from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
import logging

from bika.models import (
    Product, ProductAIInsights, ProductPriceHistory, PurchaseOrder, PurchaseOrderItem,
    InventoryTransfer, OrderItem, InventoryItem # Added InventoryItem for holding cost estimation
)

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Analyzes the financial impact of AI-driven inventory and pricing optimizations.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--period-days',
            type=int,
            default=30,
            help='Number of days back for the analysis period.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting AI-driven cost optimization analysis..."))
        
        period_days = options['period_days']
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=period_days)

        self.stdout.write(self.style.NOTICE(f"
--- Cost Optimization Analysis for Period: {start_date} to {end_date} ---"))
        
        total_revenue_impact_from_pricing = Decimal('0.00')
        total_reordering_cost = Decimal('0.00')
        total_transfer_cost = Decimal('0.00')
        total_holding_cost = Decimal('0.00')
        
        # --- Analyze Impact of Dynamic Pricing ---
        price_history_records = ProductPriceHistory.objects.filter(
            change_date__date__range=(start_date, end_date),
            reason__icontains='Automated AI adjustment'
        ).select_related('product')
        
        self.stdout.write(self.style.NOTICE("
1. Impact of Automated Pricing Adjustments:"))
        for record in price_history_records:
            # This is a very simplified revenue impact. A real one would track sales before/after change.
            # Here, we just acknowledge the change.
            self.stdout.write(f"  - Product: {record.product.name}, Price changed from {record.old_price} to {record.new_price} on {record.change_date.date()}")
            # To calculate actual revenue impact: need to fetch sales for a period before and after the price change.
            # This is beyond the scope of this simplified analysis.
            # For now, we note that price changes occurred due to AI.
        if not price_history_records.exists():
            self.stdout.write("  No automated AI price adjustments found in this period.")

        # --- Analyze Reordering Costs (from AI-driven POs) ---
        purchase_orders = PurchaseOrder.objects.filter(
            order_date__date__range=(start_date, end_date),
            status__in=['ordered', 'received'],
            ordered_by__isnull=False # Assuming automated POs are assigned to a system user
            # Alternatively, filter by a specific system user ID or type
        )
        
        # For simplicity, we assume a fixed ordering cost per PO for this analysis.
        # This could be added to Supplier or Product models.
        ORDERING_COST_PER_PO = Decimal('50.00') # Example fixed cost per purchase order
        
        for po in purchase_orders:
            total_reordering_cost += ORDERING_COST_PER_PO
            self.stdout.write(f"  - PO {po.order_number} to {po.supplier.name}: Cost {ORDERING_COST_PER_PO}")
        self.stdout.write(self.style.NOTICE(f"
2. Total Reordering Cost (Automated POs): ${total_reordering_cost:.2f} (Estimated)"))
        if not purchase_orders.exists():
            self.stdout.write("  No automated purchase orders found in this period.")

        # --- Analyze Transfer Costs (from AI-driven Transfers) ---
        inventory_transfers = InventoryTransfer.objects.filter(
            transfer_date__date__range=(start_date, end_date),
            status__in=['in_transit', 'received'],
            requested_by__isnull=False # Assuming automated transfers are assigned to a system user
            # Alternatively, filter by a specific system user ID or type
        )

        # For simplicity, assume a fixed cost per transfer item.
        TRANSFER_COST_PER_UNIT = Decimal('5.00') # Example fixed cost per unit transferred
        
        for transfer in inventory_transfers:
            transfer_cost_for_this_item = transfer.quantity * TRANSFER_COST_PER_UNIT
            total_transfer_cost += transfer_cost_for_this_item
            self.stdout.write(f"  - Transfer {transfer.transfer_number} for {transfer.product.name} ({transfer.quantity} units): Cost {transfer_cost_for_this_item}")
        self.stdout.write(self.style.NOTICE(f"
3. Total Inventory Transfer Cost (Automated Transfers): ${total_transfer_cost:.2f} (Estimated)"))
        if not inventory_transfers.exists():
            self.stdout.write("  No automated inventory transfers found in this period.")

        # --- Analyze Holding Costs (Simplified) ---
        # This is a very simplified calculation assuming average stock quantity over the period
        # A real holding cost calculation is complex (interest, obsolescence, storage space, insurance)
        
        # Get average stock quantity per product over the period (very rough estimate)
        products_with_inventory = Product.objects.filter(inventoryitem__isnull=False).distinct()
        HOLDING_COST_PER_UNIT_PER_DAY = Decimal('0.10') # Example

        self.stdout.write(self.style.NOTICE("
4. Estimated Holding Costs:"))
        for product in products_with_inventory:
            # Roughly estimate average stock for the period based on InventoryItem.quantity
            # A true calculation would need daily stock snapshots or inventory history.
            estimated_average_stock = InventoryItem.objects.filter(product=product, status='active').aggregate(Sum('quantity'))['quantity__sum'] or Decimal('0.00')
            cost_for_product = estimated_average_stock * HOLDING_COST_PER_UNIT_PER_DAY * period_days
            total_holding_cost += cost_for_product
            self.stdout.write(f"  - Product: {product.name}, Estimated Holding Cost: ${cost_for_product:.2f}")

        self.stdout.write(self.style.NOTICE(f"
Overall Estimated Total Holding Cost: ${total_holding_cost:.2f}"))

        # --- Overall Summary ---
        overall_total_cost = total_reordering_cost + total_transfer_cost + total_holding_cost
        self.stdout.write(self.style.NOTICE(f"
--- Overall Cost Optimization Summary ---"))
        self.stdout.write(self.style.SUCCESS(f"  Estimated Total Automated Action Costs: ${overall_total_cost:.2f}"))
        self.stdout.write(self.style.NOTICE(f"  Note: Revenue impact from automated pricing and potential stockout cost savings are qualitative in this simplified analysis."))

        self.stdout.write(self.style.SUCCESS("
AI-driven cost optimization analysis completed."))