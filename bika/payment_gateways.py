import requests
import base64
import json
import hashlib
import hmac
from datetime import datetime
from django.conf import settings
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class BasePaymentGateway:
    """Base class for all payment gateways"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.merchant_id = config.merchant_id
        self.environment = config.environment
    
    def make_request(self, endpoint, payload, method='POST'):
        """Make API request to payment gateway"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self.get_headers()
            
            if method.upper() == 'POST':
                response = requests.post(url, json=payload, headers=headers, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Payment gateway request error: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def get_headers(self):
        """Get request headers - to be implemented by subclasses"""
        return {'Content-Type': 'application/json'}

class MpesaGateway(BasePaymentGateway):
    """M-Pesa Tanzania Gateway"""
    
    def get_access_token(self):
        """Get M-Pesa access token"""
        try:
            url = f"{self.base_url}/oauth/v1/generate?grant_type=client_credentials"
            auth_string = f"{self.api_key}:{self.api_secret}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            
            headers = {'Authorization': f'Basic {encoded_auth}'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('access_token')
            
        except Exception as e:
            logger.error(f"M-Pesa access token error: {str(e)}")
            return None
    
    def get_timestamp(self):
        return datetime.now().strftime('%Y%m%d%H%M%S')
    
    def generate_password(self, timestamp):
        data = f"{self.merchant_id}{self.config.api_secret}{timestamp}"
        return base64.b64encode(data.encode()).decode()
    
    def stk_push(self, phone_number, amount, reference):
        """Initiate STK push"""
        try:
            access_token = self.get_access_token()
            if not access_token:
                return {'success': False, 'message': 'Failed to get access token'}
            
            timestamp = self.get_timestamp()
            password = self.generate_password(timestamp)
            
            # Format phone number
            if phone_number.startswith('0'):
                phone_number = '255' + phone_number[1:]
            elif phone_number.startswith('+'):
                phone_number = phone_number[1:]
            
            payload = {
                "BusinessShortCode": self.merchant_id,
                "Password": password,
                "Timestamp": timestamp,
                "TransactionType": "CustomerPayBillOnline",
                "Amount": amount,
                "PartyA": phone_number,
                "PartyB": self.merchant_id,
                "PhoneNumber": phone_number,
                "CallBackURL": self.config.callback_url,
                "AccountReference": reference,
                "TransactionDesc": "Bika Product Purchase"
            }
            
            url = f"{self.base_url}/mpesa/stkpush/v1/processrequest"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('ResponseCode') == '0':
                return {
                    'success': True,
                    'merchant_request_id': response_data.get('MerchantRequestID'),
                    'checkout_request_id': response_data.get('CheckoutRequestID'),
                    'message': 'Payment initiated successfully'
                }
            else:
                return {
                    'success': False,
                    'message': response_data.get('ResponseDescription', 'Payment request failed')
                }
                
        except Exception as e:
            logger.error(f"M-Pesa STK push error: {str(e)}")
            return {'success': False, 'message': str(e)}

class MTNRwandaGateway(BasePaymentGateway):
    """MTN Mobile Money Rwanda Gateway"""
    
    def request_payment(self, phone_number, amount, reference):
        """Request payment from customer"""
        try:
            # MTN Rwanda API integration
            payload = {
                "amount": str(amount),
                "currency": "RWF",
                "externalId": reference,
                "payer": {
                    "partyIdType": "MSISDN",
                    "partyId": phone_number
                },
                "payerMessage": "Payment for Bika order",
                "payeeNote": f"Order {reference}"
            }
            
            # Generate API token (implementation depends on MTN API)
            token = self.generate_api_token()
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'X-Reference-Id': reference
            }
            
            response = self.make_request('/collection/v1_0/requesttopay', payload)
            
            if response.get('status') == 'SUCCESSFUL':
                return {
                    'success': True,
                    'transaction_id': response.get('transactionRef'),
                    'message': 'Payment request sent to customer'
                }
            else:
                return {
                    'success': False,
                    'message': response.get('message', 'Payment request failed')
                }
                
        except Exception as e:
            logger.error(f"MTN Rwanda payment error: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def generate_api_token(self):
        """Generate MTN API token"""
        # Implementation for MTN API authentication
        return "mtn_api_token"

class TigoTanzaniaGateway(BasePaymentGateway):
    """Tigo Pesa Tanzania Gateway"""
    
    def initiate_payment(self, phone_number, amount, reference):
        """Initiate Tigo Pesa payment"""
        try:
            payload = {
                "msisdn": phone_number,
                "amount": str(amount),
                "currency": "TZS",
                "reference": reference,
                "description": f"Bika Order {reference}"
            }
            
            # Tigo Pesa API implementation
            headers = self.get_headers()
            response = self.make_request('/api/v1/payments', payload)
            
            if response.get('status') == 'SUCCESS':
                return {
                    'success': True,
                    'transaction_id': response.get('transactionId'),
                    'message': 'Tigo Pesa payment initiated'
                }
            else:
                return {
                    'success': False,
                    'message': response.get('message', 'Tigo Pesa payment failed')
                }
                
        except Exception as e:
            logger.error(f"Tigo Pesa payment error: {str(e)}")
            return {'success': False, 'message': str(e)}

class AirtelAfricaGateway(BasePaymentGateway):
    """Airtel Money Gateway for multiple countries"""
    
    def initiate_payment(self, phone_number, amount, reference, country):
        """Initiate Airtel Money payment"""
        try:
            payload = {
                "reference": reference,
                "subscriber": {
                    "country": country,
                    "currency": self.get_currency_for_country(country),
                    "msisdn": phone_number
                },
                "transaction": {
                    "amount": str(amount),
                    "country": country,
                    "currency": self.get_currency_for_country(country),
                    "id": reference
                }
            }
            
            headers = self.get_headers()
            response = self.make_request('/merchant/v1/payments/', payload)
            
            if response.get('status') == 'SUCCESSFUL':
                return {
                    'success': True,
                    'transaction_id': response.get('airtel_money_id'),
                    'message': 'Airtel Money payment initiated'
                }
            else:
                return {
                    'success': False,
                    'message': response.get('message', 'Airtel Money payment failed')
                }
                
        except Exception as e:
            logger.error(f"Airtel Money payment error: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def get_currency_for_country(self, country):
        currency_map = {
            'TZ': 'TZS', 'RW': 'RWF', 'UG': 'UGX', 'KE': 'KES'
        }
        return currency_map.get(country, 'USD')

class StripeGateway(BasePaymentGateway):
    """Stripe for card payments"""
    
    def create_payment_intent(self, amount, currency, payment_method_id, customer_email=None):
        """Create Stripe payment intent"""
        try:
            import stripe
            stripe.api_key = self.api_secret
            
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency.lower(),
                payment_method=payment_method_id,
                confirmation_method='manual',
                confirm=True,
                receipt_email=customer_email,
                metadata={
                    'order_reference': 'bika_order'
                }
            )
            
            return {
                'success': True,
                'payment_intent_id': intent.id,
                'client_secret': intent.client_secret,
                'status': intent.status
            }
            
        except Exception as e:
            logger.error(f"Stripe payment error: {str(e)}")
            return {'success': False, 'message': str(e)}

class PayPalGateway(BasePaymentGateway):
    """PayPal Gateway"""
    
    def create_order(self, amount, currency, return_url, cancel_url):
        """Create PayPal order"""
        try:
            payload = {
                "intent": "CAPTURE",
                "purchase_units": [
                    {
                        "amount": {
                            "currency_code": currency,
                            "value": str(amount)
                        }
                    }
                ],
                "application_context": {
                    "return_url": return_url,
                    "cancel_url": cancel_url,
                    "brand_name": "Bika Marketplace"
                }
            }
            
            headers = {
                'Authorization': f'Bearer {self.get_access_token()}',
                'Content-Type': 'application/json'
            }
            
            response = self.make_request('/v2/checkout/orders', payload)
            
            if response.get('status') == 'CREATED':
                return {
                    'success': True,
                    'order_id': response.get('id'),
                    'approval_url': next(link['href'] for link in response.get('links', []) if link['rel'] == 'approve')
                }
            else:
                return {'success': False, 'message': 'Failed to create PayPal order'}
                
        except Exception as e:
            logger.error(f"PayPal order creation error: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def get_access_token(self):
        """Get PayPal access token"""
        try:
            auth_string = f"{self.api_key}:{self.api_secret}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_auth}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            response = requests.post(
                f"{self.base_url}/v1/oauth2/token",
                headers=headers,
                data=data,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json().get('access_token')
            
        except Exception as e:
            logger.error(f"PayPal token error: {str(e)}")
            return None

class PaymentGatewayFactory:
    """Factory to create payment gateway instances"""
    
    @staticmethod
    def create_gateway(payment_method, config):
        gateway_map = {
            'mpesa': MpesaGateway,
            'mpesa_tz': MpesaGateway,
            'mpesa_ke': MpesaGateway,
            'mtn_rw': MTNRwandaGateway,
            'mtn_ug': MTNRwandaGateway,
            'tigo_tz': TigoTanzaniaGateway,
            'airtel_tz': AirtelAfricaGateway,
            'airtel_rw': AirtelAfricaGateway,
            'airtel_ug': AirtelAfricaGateway,
            'stripe': StripeGateway,
            'paypal': PayPalGateway,
        }
        
        gateway_class = gateway_map.get(payment_method)
        if gateway_class:
            return gateway_class(config)
        return None