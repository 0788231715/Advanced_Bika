from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
import logging

from bika.models import Product, ProductAIInsights, PurchaseOrder, PurchaseOrderItem, Supplier

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Simulates inventory and sales scenarios based on user-defined parameters to forecast impact.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--product-id',
            type=int,
            required=True,
            help='ID of the product for which to run the simulation.',
        )
        parser.add_argument(
            '--demand-change-percent',
            type=int,
            default=0,
            help='Percentage change in demand forecast (e.g., -10 for 10%% decrease, 20 for 20%% increase).',
        )
        parser.add_argument(
            '--lead-time-increase-days',
            type=int,
            default=0,
            help='Increase in supplier lead time in days for the product.',
        )
        parser.add_argument(
            '--simulation-duration-days',
            type=int,
            default=30,
            help='Duration of the simulation in days.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting AI-powered scenario simulation..."))
        
        product_id = options['product_id']
        demand_change_percent = Decimal(options['demand_change_percent']) / 100
        lead_time_increase_days = options['lead_time_increase_days']
        simulation_duration_days = options['simulation_duration_days']

        try:
            product = Product.objects.get(id=product_id)
        except Product.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Product with ID {product_id} not found."))
            return
        
        # Get current AI insights for the product
        try:
            ai_insight = product.ai_insights_for_product.latest('last_analyzed')
        except ProductAIInsights.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"No AI insights found for product {product.name}. Simulation cannot proceed."))
            return

        self.stdout.write(self.style.NOTICE(f"
--- Simulating Scenario for Product: {product.name} (ID: {product.id}) ---"))
        self.stdout.write(self.style.NOTICE(f"  Initial Stock: {product.stock_quantity}"))
        self.stdout.write(self.style.NOTICE(f"  Initial 7-day Demand Forecast: {ai_insight.demand_forecast_next_7_days}"))
        self.stdout.write(self.style.NOTICE(f"  Current Lead Time (assumed): 14 days")) # Placeholder for actual lead time

        # Apply scenario changes
        simulated_demand_7_day = ai_insight.demand_forecast_next_7_days * (1 + demand_change_percent)
        simulated_lead_time_days = 14 + lead_time_increase_days # Start with assumed 14 days
        
        self.stdout.write(self.style.NOTICE(f"
--- Scenario Parameters ---"))
        self.stdout.write(self.style.NOTICE(f"  Demand Change: {demand_change_percent * 100:.0f}%"))
        self.stdout.write(self.style.NOTICE(f"  Lead Time Increase: {lead_time_increase_days} days"))
        self.stdout.write(self.style.NOTICE(f"  Simulation Duration: {simulation_duration_days} days"))
        
        self.stdout.write(self.style.NOTICE(f"
--- Simulation Results ---"))
        
        # Simple daily simulation
        simulated_stock = product.stock_quantity
        total_sales_in_sim = 0
        potential_stockouts = 0
        reorders_triggered = 0
        
        # For a more robust simulation, you'd integrate the automated PO logic here
        # and track POs as they are "received" during the simulation.
        # This is a highly simplified demand-vs-stock model.
        
        for day in range(1, simulation_duration_days + 1):
            # Daily demand (simplified: average of 7-day forecast)
            daily_demand = simulated_demand_7_day / Decimal('7')
            
            # Sales for the day
            sales_today = min(simulated_stock, daily_demand)
            simulated_stock -= sales_today
            total_sales_in_sim += sales_today

            # Check for stockout
            if simulated_stock <= 0:
                potential_stockouts += 1
                self.stdout.write(self.style.WARNING(f"  Day {day}: Stockout for {product.name}!"))
            
            # Reorder logic (simplified: if stock drops below threshold, trigger PO)
            if simulated_stock <= product.low_stock_threshold and product.default_supplier and reorders_triggered == 0:
                # Assuming one reorder can cover the simulation duration
                # In a real scenario, lead time would affect when stock arrives
                reorder_qty = product.low_stock_threshold * 2 # Example reorder quantity
                self.stdout.write(self.style.INFO(f"  Day {day}: Reorder triggered for {product.name} (Qty: {reorder_qty}). Lead Time: {simulated_lead_time_days} days."))
                reorders_triggered += 1
                # If lead time is short enough, simulate receiving stock
                if simulated_lead_time_days < (simulation_duration_days - day):
                    # For simplicity, assume stock arrives after lead time
                    # More complex: stock arrives on a specific day, impacts future stock
                    pass # Stock arrival logic for simulation

            # self.stdout.write(f"  Day {day}: Stock: {simulated_stock:.0f}, Sales: {sales_today:.0f}")

        projected_final_stock = simulated_stock
        projected_revenue = total_sales_in_sim * product.price # Simplified, ignores discounts/taxes

        self.stdout.write(self.style.NOTICE(f"
--- Summary after {simulation_duration_days} days ---"))
        self.stdout.write(self.style.SUCCESS(f"  Projected Final Stock: {projected_final_stock:.0f}"))
        self.stdout.write(self.style.SUCCESS(f"  Total Simulated Sales (Units): {total_sales_in_sim:.0f}"))
        self.stdout.write(self.style.SUCCESS(f"  Estimated Revenue: ${projected_revenue:.2f}"))
        self.stdout.write(self.style.SUCCESS(f"  Days with Potential Stockouts: {potential_stockouts}"))
        self.stdout.write(self.style.SUCCESS(f"  Reorders Triggered: {reorders_triggered}"))
        
        self.stdout.write(self.style.SUCCESS("
Scenario simulation completed."))