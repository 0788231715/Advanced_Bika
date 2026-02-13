from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import logging

from bika.models import (
    ProductAlert, InventoryMovement, InventoryTransfer, Delivery, CustomUser
)
from bika.services.carrier_service import CarrierIntegrationService

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Monitors inventory movements, transfers, and deliveries for anomalies and delays, generating ProductAlerts.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--check-movement-anomalies',
            action='store_true',
            help='Check for unexpected inventory movements.',
        )
        parser.add_argument(
            '--check-transfer-delays',
            action='store_true',
            help='Check for delays in inventory transfers.',
        )
        parser.add_argument(
            '--check-delivery-delays',
            action='store_true',
            help='Check for delays in customer deliveries.',
        )
        parser.add_argument(
            '--transfer-delay-threshold-hours',
            type=int,
            default=48, # 2 days
            help='Threshold in hours after transfer_date for a pending/in_transit transfer to be considered delayed.',
        )
        parser.add_argument(
            '--delivery-delay-threshold-hours',
            type=int,
            default=24, # 1 day
            help='Threshold in hours after estimated_delivery for a non-delivered delivery to be considered delayed.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting inventory and logistics monitoring..."))
        
        system_user = CustomUser.objects.filter(is_superuser=True).first()
        if not system_user:
            self.stdout.write(self.style.ERROR("No superuser found to assign as 'detected_by'. Alerts will be generated without assigned user."))
            # For this command, we need a user to assign to 'detected_by'
            # Let's create a dummy system user if none exists for automation
            system_user, created = CustomUser.objects.get_or_create(username='system_automation', defaults={'email': 'system@bika.com', 'is_staff': True, 'is_superuser': True})
            if created:
                self.stdout.write(self.style.WARNING("Created a dummy system_automation user for alerts."))

        alert_count = 0

        # --- Monitor Inventory Movement Anomalies ---
        if options['check_movement_anomalies']:
            self.stdout.write(self.style.NOTICE("
Checking for inventory movement anomalies..."))
            
            # Simplified: Find InventoryMovement records that are not explicitly planned
            # This is complex. For now, we'll implement a conceptual check.
            # A true anomaly detection would involve comparing actual movements with expected states (e.g., item should not be here).
            
            # Placeholder: Example of detecting an item that moved locations multiple times in a short period without a clear purpose
            # This would typically require a time-window analysis of InventoryMovement
            
            # For current purposes, we will just log a placeholder message.
            self.stdout.write(self.style.WARNING("  Movement anomaly detection requires a robust 'planned movement' system. Skipping detailed check for now."))
            # if an item's current granular location (shelf, bin, pallet) doesn't match its last recorded "put_away" or "receive"
            # we could raise an alert. This implies querying InventoryItem.
            pass


        # --- Monitor Inventory Transfer Delays ---
        if options['check_transfer_delays']:
            self.stdout.write(self.style.NOTICE("
Checking for inventory transfer delays..."))
            delay_threshold = timezone.now() - timedelta(hours=options['transfer_delay_threshold_hours'])
            
            delayed_transfers = InventoryTransfer.objects.filter(
                status__in=['pending', 'in_transit'],
                transfer_date__lt=delay_threshold # Transfers started before threshold and not completed
            )

            for transfer in delayed_transfers:
                alert_message = f"Inventory Transfer TRF-{transfer.transfer_number} for {transfer.product.name} is delayed. Status: {transfer.get_status_display()} since {transfer.transfer_date.strftime('%Y-%m-%d %H:%M')}."
                
                # Check if an alert already exists for this delay
                existing_alert = ProductAlert.objects.filter(
                    alert_type='transfer_delay',
                    # Link to product if possible
                    product=transfer.product,
                    # message=alert_message, # Message can vary, so don't filter by it
                    is_resolved=False,
                    created_at__gte=timezone.now() - timedelta(days=7) # Only check recent alerts
                ).first()

                if not existing_alert:
                    ProductAlert.objects.create(
                        product=transfer.product,
                        alert_type='transfer_delay',
                        severity='high',
                        message=alert_message,
                        detected_by=system_user.username if system_user else 'System Automation',
                        # Link to delivery if it's a type of delivery
                        # For now, it's transfer specific
                    )
                    alert_count += 1
                    self.stdout.write(self.style.WARNING(f"  Generated alert: {alert_message}"))
                else:
                    self.stdout.write(f"  Alert already exists for transfer {transfer.transfer_number}.")

        # --- Monitor Delivery Delays ---
        if options['check_delivery_delays']:
            self.stdout.write(self.style.NOTICE("
Checking for customer delivery delays..."))
            delay_threshold = timezone.now() # Current time is past estimated delivery
            
            delayed_deliveries = Delivery.objects.filter(
                status__in=['pending', 'processing', 'packed', 'in_transit', 'out_for_delivery'],
                estimated_delivery__lt=delay_threshold # Estimated delivery has passed
            )
            
            carrier_service = CarrierIntegrationService() # Instantiate service

            for delivery in delayed_deliveries:
                current_tracking_status = "Unknown"
                external_tracking_url = delivery.external_tracking_url
                
                # Try to get updated tracking status from carrier service
                if delivery.carrier_name and delivery.tracking_number:
                    tracking_info = carrier_service.get_tracking_info(delivery.tracking_number, delivery.carrier_name)
                    if tracking_info:
                        current_tracking_status = tracking_info.get('status', current_tracking_status)
                        # Optionally update external_tracking_url if service provides a better one
                        if tracking_info.get('external_url'):
                            delivery.external_tracking_url = tracking_info['external_url']
                            delivery.save(update_fields=['external_tracking_url'])

                alert_message = f"Customer Delivery DEL-{delivery.delivery_number} for {delivery.client_name} is delayed. Current Status: {delivery.get_status_display()} ({current_tracking_status}). Estimated: {delivery.estimated_delivery.strftime('%Y-%m-%d %H:%M')}."
                
                # Check if an alert already exists for this delay
                existing_alert = ProductAlert.objects.filter(
                    alert_type='delivery_delay',
                    delivery=delivery, # Link alert directly to delivery
                    # message=alert_message, # Message can vary, so don't filter by it
                    is_resolved=False,
                    created_at__gte=timezone.now() - timedelta(days=7) # Only check recent alerts
                ).first()

                if not existing_alert:
                    ProductAlert.objects.create(
                        # Link to product if delivery has order items
                        product=delivery.order.items.first().product if delivery.order and delivery.order.items.first() else None,
                        delivery=delivery,
                        alert_type='delivery_delay',
                        severity='critical',
                        message=alert_message,
                        detected_by=system_user.username if system_user else 'System Automation'
                    )
                    alert_count += 1
                    self.stdout.write(self.style.WARNING(f"  Generated alert: {alert_message}"))
                else:
                    self.stdout.write(f"  Alert already exists for delivery {delivery.delivery_number}.")

        self.stdout.write(self.style.SUCCESS(
            f"
Finished inventory and logistics monitoring. Generated {alert_count} new alerts."
        ))