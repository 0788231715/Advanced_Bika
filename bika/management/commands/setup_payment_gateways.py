# bika/management/commands/setup_payment_gateways.py
from django.core.management.base import BaseCommand
from bika.models import PaymentGatewaySettings

class Command(BaseCommand):
    help = 'Setup default payment gateway configurations'
    
    def handle(self, *args, **kwargs):
        # Tanzania gateways
        gateways = [
            {
                'gateway': 'mpesa_tz',
                'display_name': 'M-Pesa Tanzania',
                'supported_countries': ['TZ'],
                'supported_currencies': ['TZS'],
                'is_active': True,
                'environment': 'sandbox',
                'transaction_fee_percent': 0.5,
                'transaction_fee_fixed': 0,
            },
            {
                'gateway': 'tigo_tz',
                'display_name': 'Tigo Pesa Tanzania',
                'supported_countries': ['TZ'],
                'supported_currencies': ['TZS'],
                'is_active': True,
                'environment': 'sandbox',
                'transaction_fee_percent': 0.5,
                'transaction_fee_fixed': 0,
            },
            {
                'gateway': 'airtel_tz',
                'display_name': 'Airtel Money Tanzania',
                'supported_countries': ['TZ'],
                'supported_currencies': ['TZS'],
                'is_active': True,
                'environment': 'sandbox',
                'transaction_fee_percent': 0.5,
                'transaction_fee_fixed': 0,
            },
            # Add more gateways as needed
        ]
        
        for gateway_data in gateways:
            gateway, created = PaymentGatewaySettings.objects.update_or_create(
                gateway=gateway_data['gateway'],
                defaults=gateway_data
            )
            
            status = "Created" if created else "Updated"
            self.stdout.write(
                self.style.SUCCESS(f'{status} {gateway.display_name}')
            )