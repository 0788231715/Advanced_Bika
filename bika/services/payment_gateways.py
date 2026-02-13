# bika/services/payment_gateways.py
import logging
import json
import requests
import hashlib
import hmac
import base64
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from django.urls import reverse
from ..models import Payment, PaymentGatewaySettings

logger = logging.getLogger(__name__)

# ==================== BASE PAYMENT GATEWAY ====================

class BasePaymentGateway:
    """Base class for all payment gateways"""
    
    def __init__(self, gateway_config):
        self.gateway_config = gateway_config
        self.gateway_name = gateway_config.gateway
        self.is_active = gateway_config.is_active
        self.environment = gateway_config.environment
        self.base_url = gateway_config.base_url
        self.callback_url = gateway_config.callback_url
        self.api_key = gateway_config.api_key
        self.api_secret = gateway_config.api_secret
        self.merchant_id = gateway_config.merchant_id
    
    def is_available(self):
        """Check if gateway is available for use"""
        return self.is_active and bool(self.api_key) and bool(self.api_secret)
    
    def generate_signature(self, data, secret_key=None):
        """Generate HMAC signature for secure requests"""
        secret = secret_key or self.api_secret
        message = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def validate_webhook(self, payload, signature):
        """Validate webhook signature"""
        generated_signature = self.generate_signature(payload)
        return hmac.compare_digest(generated_signature, signature)
    
    def format_amount(self, amount, currency='TZS'):
        """Format amount according to gateway requirements"""
        # Different gateways have different requirements
        if currency in ['TZS', 'RWF', 'UGX', 'KES']:
            # East African currencies - no decimal places
            return int(amount)
        else:
            # International currencies - keep decimal places
            return float(amount)
    
    def get_supported_countries(self):
        """Get list of supported countries"""
        return self.gateway_config.supported_countries or []
    
    def get_supported_currencies(self):
        """Get list of supported currencies"""
        return self.gateway_config.supported_currencies or []
    
    def calculate_fee(self, amount):
        """Calculate transaction fees"""
        percent_fee = (amount * self.gateway_config.transaction_fee_percent) / 100
        total_fee = percent_fee + self.gateway_config.transaction_fee_fixed
        return total_fee
    
    # Abstract methods to be implemented by subclasses
    def initiate_payment(self, payment):
        """Initiate payment - to be implemented by subclasses"""
        raise NotImplementedError
    
    def verify_payment(self, transaction_id):
        """Verify payment status - to be implemented by subclasses"""
        raise NotImplementedError
    
    def process_webhook(self, payload):
        """Process webhook - to be implemented by subclasses"""
        raise NotImplementedError
    
    def refund_payment(self, transaction_id, amount, reason=""):
        """Refund payment - to be implemented by subclasses"""
        raise NotImplementedError

# ==================== TANZANIA PAYMENT GATEWAYS ====================

class MPesaGateway(BasePaymentGateway):
    """M-Pesa Tanzania Payment Gateway"""
    
    def __init__(self, gateway_config):
        super().__init__(gateway_config)
        # M-Pesa specific endpoints
        if self.environment == 'sandbox':
            self.base_url = "https://sandbox.safaricom.co.ke"
        else:
            self.base_url = "https://api.safaricom.co.ke"
    
    def initiate_payment(self, payment):
        """Initiate M-Pesa STK Push"""
        try:
            # Get access token
            token_response = self.get_access_token()
            if not token_response.get('success'):
                return token_response
            
            access_token = token_response['access_token']
            
            # Prepare STK Push request
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            password = base64.b64encode(
                f"{self.gateway_config.merchant_id}{self.gateway_config.api_secret}{timestamp}".encode()
            ).decode()
            
            stk_push_data = {
                "BusinessShortCode": self.gateway_config.merchant_id,
                "Password": password,
                "Timestamp": timestamp,
                "TransactionType": "CustomerPayBillOnline",
                "Amount": self.format_amount(payment.amount, payment.currency),
                "PartyA": payment.mobile_money_phone,
                "PartyB": self.gateway_config.merchant_id,
                "PhoneNumber": payment.mobile_money_phone,
                "CallBackURL": f"{self.callback_url}?payment_id={payment.id}",
                "AccountReference": f"ORDER-{payment.order.order_number}",
                "TransactionDesc": f"Payment for order #{payment.order.order_number}"
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/mpesa/stkpush/v1/processrequest",
                json=stk_push_data,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('ResponseCode') == '0':
                # Success
                payment.transaction_id = response_data.get('CheckoutRequestID')
                payment.mobile_money_provider = 'mpesa_tz'
                payment.save()
                
                return {
                    'success': True,
                    'transaction_id': response_data.get('CheckoutRequestID'),
                    'message': 'Payment initiated successfully',
                    'checkout_request_id': response_data.get('CheckoutRequestID'),
                    'customer_message': response_data.get('CustomerMessage', '')
                }
            else:
                error_msg = response_data.get('errorMessage') or response_data.get('ResponseDescription') or 'Payment initiation failed'
                return {
                    'success': False,
                    'error': error_msg,
                    'response_code': response_data.get('ResponseCode')
                }
                
        except Exception as e:
            logger.error(f"M-Pesa payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_access_token(self):
        """Get M-Pesa access token"""
        try:
            auth_string = f"{self.api_key}:{self.api_secret}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_auth}'
            }
            
            response = requests.get(
                f"{self.base_url}/oauth/v1/generate?grant_type=client_credentials",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'access_token': data.get('access_token'),
                    'expires_in': data.get('expires_in')
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get access token'
                }
                
        except Exception as e:
            logger.error(f"M-Pesa access token error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_payment(self, transaction_id):
        """Verify M-Pesa payment status"""
        try:
            token_response = self.get_access_token()
            if not token_response.get('success'):
                return token_response
            
            access_token = token_response['access_token']
            
            verify_data = {
                "BusinessShortCode": self.gateway_config.merchant_id,
                "Password": base64.b64encode(
                    f"{self.gateway_config.merchant_id}{self.gateway_config.api_secret}{datetime.now().strftime('%Y%m%d%H%M%S')}".encode()
                ).decode(),
                "Timestamp": datetime.now().strftime('%Y%m%d%H%M%S'),
                "CheckoutRequestID": transaction_id
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/mpesa/stkpushquery/v1/query",
                json=verify_data,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 200:
                result_code = response_data.get('ResultCode')
                result_desc = response_data.get('ResultDesc')
                
                if result_code == '0':
                    return {
                        'success': True,
                        'status': 'completed',
                        'transaction_id': response_data.get('TransactionID'),
                        'amount': response_data.get('Amount'),
                        'phone_number': response_data.get('PhoneNumber'),
                        'message': result_desc
                    }
                elif result_code == '1032':
                    return {
                        'success': True,
                        'status': 'cancelled',
                        'message': 'Payment cancelled by user'
                    }
                else:
                    return {
                        'success': True,
                        'status': 'failed',
                        'message': result_desc
                    }
            else:
                return {
                    'success': False,
                    'error': 'Verification request failed'
                }
                
        except Exception as e:
            logger.error(f"M-Pesa verification error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_webhook(self, payload):
        """Process M-Pesa webhook callback"""
        try:
            # M-Pesa sends callback data
            callback_data = payload.get('Body', {}).get('stkCallback', {})
            checkout_request_id = callback_data.get('CheckoutRequestID')
            result_code = callback_data.get('ResultCode')
            result_desc = callback_data.get('ResultDesc')
            
            if result_code == '0':
                # Payment successful
                callback_metadata = callback_data.get('CallbackMetadata', {}).get('Item', [])
                amount = None
                mpesa_receipt_number = None
                phone_number = None
                
                for item in callback_metadata:
                    if item.get('Name') == 'Amount':
                        amount = item.get('Value')
                    elif item.get('Name') == 'MpesaReceiptNumber':
                        mpesa_receipt_number = item.get('Value')
                    elif item.get('Name') == 'PhoneNumber':
                        phone_number = item.get('Value')
                
                return {
                    'success': True,
                    'transaction_id': mpesa_receipt_number,
                    'checkout_request_id': checkout_request_id,
                    'amount': amount,
                    'phone_number': phone_number,
                    'status': 'completed',
                    'message': result_desc
                }
            else:
                # Payment failed
                return {
                    'success': True,
                    'checkout_request_id': checkout_request_id,
                    'status': 'failed',
                    'message': result_desc
                }
                
        except Exception as e:
            logger.error(f"M-Pesa webhook processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def refund_payment(self, transaction_id, amount, reason=""):
        """Refund M-Pesa payment"""
        # M-Pesa refunds require approval from Safaricom
        # This is a simplified version
        try:
            token_response = self.get_access_token()
            if not token_response.get('success'):
                return token_response
            
            access_token = token_response['access_token']
            
            refund_data = {
                "CommandID": "TransactionReversal",
                "ReceiverParty": self.gateway_config.merchant_id,
                "RecieverIdentifierType": "11",
                "Remarks": reason or "Refund requested",
                "Amount": self.format_amount(amount, 'TZS'),
                "Initiator": self.gateway_config.merchant_id,
                "SecurityCredential": self.generate_security_credential(),
                "QueueTimeOutURL": f"{self.callback_url}/refund-timeout",
                "ResultURL": f"{self.callback_url}/refund-result",
                "TransactionID": transaction_id,
                "Occasion": "Refund"
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/mpesa/reversal/v1/request",
                json=refund_data,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('ResponseCode') == '0':
                return {
                    'success': True,
                    'transaction_id': response_data.get('TransactionID'),
                    'conversation_id': response_data.get('ConversationID'),
                    'message': 'Refund initiated successfully'
                }
            else:
                return {
                    'success': False,
                    'error': response_data.get('ResponseDescription', 'Refund failed')
                }
                
        except Exception as e:
            logger.error(f"M-Pesa refund error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_security_credential(self):
        """Generate M-Pesa security credential"""
        # This is a simplified version
        # In production, use proper encryption
        import hashlib
        from Crypto.Cipher import PKCS1_v1_5
        from Crypto.PublicKey import RSA
        import base64
        
        # Load public key and encrypt
        # This is a placeholder - actual implementation requires Safaricom's public key
        return base64.b64encode(hashlib.sha256(self.api_secret.encode()).digest()).decode()

class TigoPesaGateway(BasePaymentGateway):
    """Tigo Pesa Tanzania Payment Gateway"""
    
    def initiate_payment(self, payment):
        """Initiate Tigo Pesa payment"""
        try:
            # Tigo Pesa API endpoint
            url = f"{self.base_url}/payment/request" if self.environment == 'live' else f"{self.base_url}/sandbox/payment/request"
            
            payment_data = {
                "msisdn": payment.mobile_money_phone,
                "amount": str(self.format_amount(payment.amount, payment.currency)),
                "reference": f"ORDER-{payment.order.order_number}",
                "narration": f"Payment for order #{payment.order.order_number}",
                "callback_url": f"{self.callback_url}?payment_id={payment.id}"
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                url,
                json=payment_data,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('status') == 'success':
                payment.transaction_id = response_data.get('transaction_id')
                payment.mobile_money_provider = 'tigo_tz'
                payment.save()
                
                return {
                    'success': True,
                    'transaction_id': response_data.get('transaction_id'),
                    'message': 'Payment initiated successfully'
                }
            else:
                return {
                    'success': False,
                    'error': response_data.get('message', 'Payment initiation failed')
                }
                
        except Exception as e:
            logger.error(f"Tigo Pesa payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_payment(self, transaction_id):
        """Verify Tigo Pesa payment"""
        try:
            url = f"{self.base_url}/payment/status/{transaction_id}" if self.environment == 'live' else f"{self.base_url}/sandbox/payment/status/{transaction_id}"
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response_data = response.json()
            
            if response.status_code == 200:
                status = response_data.get('status')
                if status == 'SUCCESS':
                    return {
                        'success': True,
                        'status': 'completed',
                        'transaction_id': transaction_id,
                        'amount': response_data.get('amount'),
                        'message': 'Payment completed successfully'
                    }
                elif status in ['PENDING', 'PROCESSING']:
                    return {
                        'success': True,
                        'status': 'pending',
                        'message': 'Payment is being processed'
                    }
                else:
                    return {
                        'success': True,
                        'status': 'failed',
                        'message': response_data.get('message', 'Payment failed')
                    }
            else:
                return {
                    'success': False,
                    'error': 'Verification request failed'
                }
                
        except Exception as e:
            logger.error(f"Tigo Pesa verification error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class AirtelMoneyGateway(BasePaymentGateway):
    """Airtel Money Tanzania Payment Gateway"""
    
    def initiate_payment(self, payment):
        """Initiate Airtel Money payment"""
        try:
            # Generate unique transaction ID
            import uuid
            transaction_id = f"AIRTEL-{uuid.uuid4().hex[:12].upper()}"
            
            # Airtel Money typically uses USSD push
            # This is a simplified simulation
            payment.transaction_id = transaction_id
            payment.mobile_money_provider = 'airtel_tz'
            payment.save()
            
            # In production, you would call Airtel's API here
            return {
                'success': True,
                'transaction_id': transaction_id,
                'message': 'Payment request sent to Airtel Money',
                'instruction': f"Check your Airtel Money on {payment.mobile_money_phone} to complete payment"
            }
                
        except Exception as e:
            logger.error(f"Airtel Money payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_payment(self, transaction_id):
        """Verify Airtel Money payment"""
        # Simplified verification - in production, call Airtel's API
        return {
            'success': True,
            'status': 'pending',  # Always pending in simulation
            'transaction_id': transaction_id,
            'message': 'Verification would be done via Airtel API'
        }

# ==================== RWANDA PAYMENT GATEWAYS ====================

class MTNRwandaGateway(BasePaymentGateway):
    """MTN Rwanda Mobile Money Gateway - Production-ready structure"""
    
    def __init__(self, gateway_config):
        super().__init__(gateway_config)
        # MTN specific endpoints - User MUST configure these
        # These URLs are illustrative. Replace with actual MTN MoMo API URLs.
        if self.environment == 'sandbox':
            self.api_base_url = "https://sandbox.mtn.com/api" # Placeholder: Replace with actual MTN Sandbox API URL
            self.token_url = "https://sandbox.mtn.com/oauth/token" # Placeholder: Replace with actual MTN Token URL
            self.initiate_url = "https://sandbox.mtn.com/collection/v1_0/requesttopay" # Placeholder: Replace with actual MTN Request to Pay URL
            self.status_url = "https://sandbox.mtn.com/collection/v1_0/requesttopay/" # Placeholder: Replace with actual MTN Request to Pay Status URL
        else:
            self.api_base_url = "https://api.mtn.com/api" # Placeholder: Replace with actual MTN Live API URL
            self.token_url = "https://api.mtn.com/oauth/token" # Placeholder: Replace with actual MTN Token URL
            self.initiate_url = "https://api.mtn.com/collection/v1_0/requesttopay" # Placeholder: Replace with actual MTN Request to Pay URL
            self.status_url = "https://api.mtn.com/collection/v1_0/requesttopay/" # Placeholder: Replace with actual MTN Request to Pay Status URL
        
        # Additional MTN specific credentials from gateway_config (these fields need to be added to PaymentGatewaySettings model)
        # For MTN MoMo API, you typically need:
        # 1. Primary Key (Ocp-Apim-Subscription-Key)
        # 2. Secondary Key (for basic auth in token)
        # 3. User ID (often UUID for API User)
        # 4. API Key (for API User)
        
        # NOTE: The PaymentGatewaySettings model needs to be extended to store `api_user` and `api_password`
        # for MTN MoMo specific API User creation. For now, using existing api_key/api_secret.
        
        self.primary_key = gateway_config.api_key # Ocp-Apim-Subscription-Key, also used for basic auth username
        self.secondary_key = gateway_config.api_secret # Basic auth password
        self.api_user_uuid = gateway_config.merchant_id # This typically corresponds to x_reference_id for the API User in MTN
        self.api_user_key = gateway_config.webhook_secret # Re-using webhook_secret for API User's API Key
        
        # Your publicly accessible domain for webhooks. User MUST configure this in settings.py.
        self.callback_host = getattr(settings, 'MTN_CALLBACK_HOST', 'https://your-public-domain.com')

    def get_access_token(self):
        """Get MTN API access token (for Collection API)"""
        try:
            # First, create an API User if not already created (this is usually a one-time setup step)
            # This logic needs to be handled outside, or in an admin command, not on every token request.
            # For simplicity here, we assume the API User is pre-created and we have its UUID and API Key.
            
            # Requesting token using API User's UUID and API Key
            # This part assumes a specific MTN API token endpoint flow. Adjust as per actual API.
            headers = {
                'Authorization': f'Basic {base64.b64encode(f"{self.api_user_uuid}:{self.api_user_key}".encode()).decode()}',
                'Ocp-Apim-Subscription-Key': self.primary_key,
                'Content-Type': 'application/json'
            }
            data = {'grant_type': 'client_credentials'}

            response = requests.post(self.token_url, headers=headers, json=data, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors
            token_data = response.json()
            
            return {
                'success': True,
                'access_token': token_data.get('access_token'),
                'expires_in': token_data.get('expires_in')
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"MTN Token request failed: {e}")
            return {'success': False, 'error': f"MTN Token request failed: {e}"}
        except json.JSONDecodeError:
            logger.error(f"MTN Token: Invalid JSON response: {getattr(response, 'text', 'N/A')}")
            return {'success': False, 'error': "MTN Token: Invalid API response"}
        except Exception as e:
            logger.error(f"MTN Token unexpected error: {e}")
            return {'success': False, 'error': f"MTN Token unexpected error: {e}"}

    def initiate_payment(self, payment):
        """Initiate MTN Mobile Money payment (Request to Pay)"""
        import uuid # Ensure uuid is imported at the top of the file
        try:
            token_response = self.get_access_token()
            if not token_response['success']:
                return token_response
            access_token = token_response['access_token']

            # Generate a UUID for the transaction reference
            transaction_uuid = str(uuid.uuid4())

            headers = {
                'Authorization': f'Bearer {access_token}',
                'X-Reference-Id': transaction_uuid, # This is a unique ID for the request
                'X-Target-Environment': self.environment.upper(), # e.g., "SANDBOX" or "PRODUCTION"
                'Content-Type': 'application/json',
                'Ocp-Apim-Subscription-Key': self.primary_key,
                # The X-Callback-Url must be publicly accessible and configured in MTN Developer Portal
                'X-Callback-Url': f"{self.callback_host}{reverse('bika:payment_webhook_mtn')}" # Specific webhook for MTN
            }
            
            data = {
                "amount": str(int(payment.amount)), # MTN typically uses integer amount (e.g., 1000 for 10.00 if currency has 2 decimal places)
                "currency": payment.currency.upper(),
                "externalId": f"ORDER-{payment.order.order_number}", # Your own reference
                "payer": {
                    "partyIdType": "MSISDN", # Mobile Subscriber ISDN
                    "partyId": payment.mobile_money_phone # User's mobile money number (e.g., "25078XXXXXXX")
                },
                "payerMessage": f"Payment for order {payment.order.order_number}",
                "payeeNote": f"Payment for order {payment.order.order_number}"
            }

            response = requests.post(self.initiate_url, headers=headers, json=data, timeout=30)
            
            # MTN's request to pay often returns 202 Accepted with no body on success, but an error can return a body.
            response.raise_for_status() # Will raise an exception for 4xx/5xx responses
            
            if response.status_code == 202:
                payment.transaction_id = transaction_uuid # Store MTN's reference ID (X-Reference-Id)
                payment.mobile_money_provider = 'mtn_rw' # Use configured gateway name
                payment.save()

                return {
                    'success': True,
                    'transaction_id': transaction_uuid,
                    'message': 'MTN Mobile Money payment request sent. Awaiting confirmation from user.',
                    'instruction': f"Confirm payment on your MTN Mobile Money account. Transaction ID: {transaction_uuid}"
                }
            else:
                # This block might be reached if response.raise_for_status() doesn't catch all cases
                error_msg = response.text or 'MTN payment initiation failed with unknown error'
                logger.error(f"MTN Payment Initiation Error ({response.status_code}): {error_msg}")
                return {'success': False, 'error': error_msg}

        except requests.exceptions.RequestException as e:
            logger.error(f"MTN Payment initiation request failed: {e}. Response: {getattr(e.response, 'text', 'N/A')}")
            return {'success': False, 'error': f"MTN Payment initiation request failed: {e}"}
        except Exception as e:
            logger.error(f"MTN Payment initiation unexpected error: {e}")
            return {'success': False, 'error': f"MTN Payment initiation unexpected error: {e}"}

    def verify_payment(self, transaction_id):
        """Verify MTN Mobile Money payment status using the X-Reference-Id"""
        try:
            token_response = self.get_access_token()
            if not token_response['success']:
                return token_response
            access_token = token_response['access_token']

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Ocp-Apim-Subscription-Key': self.primary_key,
                'X-Target-Environment': self.environment.upper(),
            }

            # transaction_id here is MTN's X-Reference-Id (UUID)
            response = requests.get(f"{self.status_url}{transaction_id}", headers=headers, timeout=30)
            response.raise_for_status()
            status_data = response.json()

            mtn_status = status_data.get('status')
            
            status_map = {
                "PENDING": "pending",
                "SUCCESSFUL": "completed",
                "FAILED": "failed"
            }
            
            payment_status = status_map.get(mtn_status, "pending")

            return {
                'success': True,
                'status': payment_status,
                'transaction_id': transaction_id, # This is the X-Reference-Id
                'amount': float(status_data.get('amount', 0)),
                'currency': status_data.get('currency'),
                'message': f"MTN status: {mtn_status}",
                'financial_transaction_id': status_data.get('financialTransactionId') # Actual transaction ID from MTN
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"MTN Payment verification request failed: {e}. Response: {getattr(e.response, 'text', 'N/A')}")
            return {'success': False, 'error': f"MTN Payment verification request failed: {e}"}
        except json.JSONDecodeError:
            logger.error(f"MTN Verification: Invalid JSON response: {getattr(response, 'text', 'N/A')}")
            return {'success': False, 'error': "MTN Verification: Invalid API response"}
        except Exception as e:
            logger.error(f"MTN Payment verification unexpected error: {e}")
            return {'success': False, 'error': f"MTN Payment verification unexpected error: {e}"}

    def process_webhook(self, payload):
        """Process MTN Mobile Money webhook callback"""
        try:
            # MTN webhooks typically provide the status update for a transaction X-Reference-Id
            # The structure of the payload depends on MTN's webhook implementation
            # This is an example structure based on common MoMo webhook designs.
            
            transaction_uuid = payload.get('referenceId') # This maps to our payment.transaction_id
            mtn_status = payload.get('status')
            
            status_map = {
                "PENDING": "pending",
                "SUCCESSFUL": "completed",
                "FAILED": "failed"
            }
            payment_status = status_map.get(mtn_status, "failed")

            return {
                'success': True,
                'transaction_id': transaction_uuid,
                'status': payment_status,
                'message': f"Webhook update from MTN: {mtn_status}",
                'raw_payload': payload
            }
        except Exception as e:
            logger.error(f"MTN webhook processing error: {e}")
            return {'success': False, 'error': f"MTN webhook processing error: {e}"}

# ==================== INTERNATIONAL PAYMENT GATEWAYS ====================

class StripeGateway(BasePaymentGateway):
    """Stripe Payment Gateway"""
    
    def __init__(self, gateway_config):
        super().__init__(gateway_config)
        import stripe
        self.stripe = stripe
        self.stripe.api_key = self.api_key
    
    def initiate_payment(self, payment):
        """Create Stripe payment intent"""
        try:
            # Convert amount to cents for Stripe
            amount_in_cents = int(payment.amount * 100)
            
            # Create payment intent
            intent = self.stripe.PaymentIntent.create(
                amount=amount_in_cents,
                currency=payment.currency.lower(),
                metadata={
                    'order_id': payment.order.id,
                    'order_number': payment.order.order_number,
                    'payment_id': payment.id
                },
                description=f"Order #{payment.order.order_number}",
                # For card payments, you might want to add card details here
            )
            
            payment.transaction_id = intent.id
            payment.save()
            
            return {
                'success': True,
                'transaction_id': intent.id,
                'client_secret': intent.client_secret,
                'message': 'Stripe payment intent created',
                'requires_action': intent.status == 'requires_action'
            }
                
        except self.stripe.error.StripeError as e:
            logger.error(f"Stripe payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Stripe payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_payment(self, transaction_id):
        """Verify Stripe payment"""
        try:
            intent = self.stripe.PaymentIntent.retrieve(transaction_id)
            
            status_map = {
                'succeeded': 'completed',
                'processing': 'pending',
                'requires_action': 'pending',
                'requires_payment_method': 'failed',
                'canceled': 'cancelled'
            }
            
            return {
                'success': True,
                'status': status_map.get(intent.status, 'pending'),
                'transaction_id': transaction_id,
                'amount': intent.amount / 100,  # Convert from cents
                'currency': intent.currency.upper(),
                'message': f"Payment {intent.status}"
            }
                
        except self.stripe.error.StripeError as e:
            logger.error(f"Stripe verification error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class PayPalGateway(BasePaymentGateway):
    """PayPal Payment Gateway"""
    
    def initiate_payment(self, payment):
        """Create PayPal payment"""
        try:
            # PayPal API endpoints
            if self.environment == 'sandbox':
                api_url = "https://api.sandbox.paypal.com"
            else:
                api_url = "https://api.paypal.com"
            
            # Get access token
            auth_response = requests.post(
                f"{api_url}/v1/oauth2/token",
                auth=(self.api_key, self.api_secret),
                data={'grant_type': 'client_credentials'},
                headers={'Accept': 'application/json', 'Accept-Language': 'en_US'},
                timeout=10
            )
            
            if auth_response.status_code != 200:
                return {
                    'success': False,
                    'error': 'Failed to get PayPal access token'
                }
            
            access_token = auth_response.json().get('access_token')
            
            # Create payment
            payment_data = {
                "intent": "sale",
                "payer": {
                    "payment_method": "paypal"
                },
                "transactions": [{
                    "amount": {
                        "total": str(payment.amount),
                        "currency": payment.currency
                    },
                    "description": f"Payment for order #{payment.order.order_number}",
                    "custom": f"ORDER-{payment.order.order_number}"
                }],
                "redirect_urls": {
                    "return_url": f"{self.callback_url}?payment_id={payment.id}&status=success",
                    "cancel_url": f"{self.callback_url}?payment_id={payment.id}&status=cancelled"
                }
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{api_url}/v1/payments/payment",
                json=payment_data,
                headers=headers,
                timeout=30
            )
            
            response_data = response.json()
            
            if response.status_code == 201:
                payment.transaction_id = response_data.get('id')
                payment.save()
                
                # Get approval URL
                approval_url = None
                for link in response_data.get('links', []):
                    if link.get('rel') == 'approval_url':
                        approval_url = link.get('href')
                        break
                
                return {
                    'success': True,
                    'transaction_id': response_data.get('id'),
                    'approval_url': approval_url,
                    'message': 'PayPal payment created'
                }
            else:
                return {
                    'success': False,
                    'error': response_data.get('message', 'PayPal payment creation failed')
                }
                
        except Exception as e:
            logger.error(f"PayPal payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# ==================== PAYMENT GATEWAY FACTORY ====================

class PaymentGatewayFactory:
    """Factory class to create payment gateway instances"""
    
    @staticmethod
    def create_gateway(gateway_name):
        """Create a payment gateway instance based on name"""
        
        # Get gateway configuration from database
        try:
            gateway_config = PaymentGatewaySettings.objects.get(gateway=gateway_name)
        except PaymentGatewaySettings.DoesNotExist:
            logger.error(f"Payment gateway configuration not found: {gateway_name}")
            return None
        
        # Map gateway names to classes
        gateway_classes = {
            # Tanzania
            'mpesa_tz': MPesaGateway,
            'tigo_tz': TigoPesaGateway,
            'airtel_tz': AirtelMoneyGateway,
            'halotel_tz': AirtelMoneyGateway,  # Using Airtel as base
            
            # Rwanda
            'mtn_rw': MTNRwandaGateway,
            'airtel_rw': AirtelMoneyGateway,
            
            # Uganda
            'mtn_ug': MTNRwandaGateway,  # Using MTN as base
            'airtel_ug': AirtelMoneyGateway,
            
            # Kenya
            'mpesa_ke': MPesaGateway,
            
            # International
            'stripe': StripeGateway,
            'paypal': PayPalGateway,
        }
        
        gateway_class = gateway_classes.get(gateway_name)
        
        if not gateway_class:
            logger.error(f"Payment gateway class not found: {gateway_name}")
            return None
        
        try:
            gateway = gateway_class(gateway_config)
            if gateway.is_available():
                return gateway
            else:
                logger.warning(f"Payment gateway {gateway_name} is not properly configured")
                return None
        except Exception as e:
            logger.error(f"Error creating payment gateway {gateway_name}: {e}")
            return None
    
    @staticmethod
    def get_available_gateways():
        """Get list of available payment gateways"""
        available_gateways = []
        
        # Get all active gateway configurations
        gateway_configs = PaymentGatewaySettings.objects.filter(is_active=True)
        
        for config in gateway_configs:
            gateway = PaymentGatewayFactory.create_gateway(config.gateway)
            if gateway:
                available_gateways.append({
                    'gateway': config.gateway,
                    'display_name': config.display_name or config.get_gateway_display(),
                    'supported_countries': config.supported_countries,
                    'supported_currencies': config.supported_currencies,
                    'instance': gateway
                })
        
        return available_gateways
    
    @staticmethod
    def get_gateway_for_payment(payment):
        """Get appropriate gateway for a payment"""
        payment_method = payment.payment_method
        
        # Map payment methods to gateway names
        method_to_gateway = {
            # Tanzania
            'mpesa': 'mpesa_tz',
            'tigo_tz': 'tigo_tz',
            'airtel_tz': 'airtel_tz',
            'halotel_tz': 'halotel_tz',
            
            # Rwanda
            'mtn_rw': 'mtn_rw',
            'airtel_rw': 'airtel_rw',
            
            # Uganda
            'mtn_ug': 'mtn_ug',
            'airtel_ug': 'airtel_ug',
            
            # Kenya
            'mpesa_ke': 'mpesa_ke',
            
            # International
            'visa': 'stripe',
            'mastercard': 'stripe',
            'amex': 'stripe',
            'paypal': 'paypal',
        }
        
        gateway_name = method_to_gateway.get(payment_method)
        
        if not gateway_name:
            # Try to infer from payment method
            if payment_method.startswith('mpesa'):
                gateway_name = 'mpesa_tz'
            elif payment_method.startswith('tigo'):
                gateway_name = 'tigo_tz'
            elif payment_method.startswith('airtel'):
                gateway_name = 'airtel_tz'
            elif payment_method in ['visa', 'mastercard', 'amex']:
                gateway_name = 'stripe'
            elif payment_method == 'paypal':
                gateway_name = 'paypal'
        
        if gateway_name:
            return PaymentGatewayFactory.create_gateway(gateway_name)
        
        return None

# ==================== PAYMENT SERVICE ====================

class PaymentService:
    """Main payment service that handles all payment operations"""
    
    def __init__(self):
        self.factory = PaymentGatewayFactory()
    
    def initiate_payment(self, payment):
        """Initiate a payment"""
        gateway = self.factory.get_gateway_for_payment(payment)
        
        if not gateway:
            return {
                'success': False,
                'error': f'No payment gateway available for {payment.payment_method}'
            }
        
        try:
            result = gateway.initiate_payment(payment)
            
            # Update payment status based on result
            if result.get('success'):
                payment.status = 'processing'
                if result.get('transaction_id'):
                    payment.transaction_id = result['transaction_id']
                payment.save()
            
            return result
            
        except Exception as e:
            logger.error(f"Payment initiation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_payment(self, payment):
        """Verify a payment status"""
        if not payment.transaction_id:
            return {
                'success': False,
                'error': 'No transaction ID available'
            }
        
        gateway = self.factory.get_gateway_for_payment(payment)
        
        if not gateway:
            return {
                'success': False,
                'error': f'No payment gateway available for {payment.payment_method}'
            }
        
        try:
            result = gateway.verify_payment(payment.transaction_id)
            
            # Update payment status based on verification result
            if result.get('success'):
                status = result.get('status')
                if status == 'completed':
                    payment.status = 'completed'
                    payment.paid_at = timezone.now()
                elif status == 'failed':
                    payment.status = 'failed'
                elif status == 'cancelled':
                    payment.status = 'cancelled'
                
                payment.save()
            
            return result
            
        except Exception as e:
            logger.error(f"Payment verification error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_webhook(self, gateway_name, payload, signature=None):
        """Process payment webhook"""
        gateway = self.factory.create_gateway(gateway_name)
        
        if not gateway:
            return {
                'success': False,
                'error': f'Payment gateway not found: {gateway_name}'
            }
        
        # Validate webhook signature if provided
        if signature and gateway.gateway_config.webhook_secret:
            if not gateway.validate_webhook(payload, signature):
                return {
                    'success': False,
                    'error': 'Invalid webhook signature'
                }
        
        try:
            result = gateway.process_webhook(payload)
            
            # Find and update the corresponding payment
            if result.get('success') and result.get('transaction_id'):
                transaction_id = result['transaction_id']
                
                try:
                    payment = Payment.objects.get(transaction_id=transaction_id)
                    status = result.get('status')
                    
                    if status == 'completed':
                        payment.status = 'completed'
                        payment.paid_at = timezone.now()
                        # Update order status
                        payment.order.status = 'confirmed'
                        payment.order.save()
                    elif status == 'failed':
                        payment.status = 'failed'
                    elif status == 'cancelled':
                        payment.status = 'cancelled'
                    
                    payment.save()
                    
                    result['payment_id'] = payment.id
                    result['order_id'] = payment.order.id
                    
                except Payment.DoesNotExist:
                    logger.warning(f"Payment not found for transaction: {transaction_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def refund_payment(self, payment, amount=None, reason=""):
        """Refund a payment"""
        if not payment.transaction_id:
            return {
                'success': False,
                'error': 'No transaction ID available for refund'
            }
        
        gateway = self.factory.get_gateway_for_payment(payment)
        
        if not gateway:
            return {
                'success': False,
                'error': f'No payment gateway available for refund'
            }
        
        refund_amount = amount or payment.amount
        
        try:
            result = gateway.refund_payment(payment.transaction_id, refund_amount, reason)
            
            if result.get('success'):
                payment.status = 'refunded'
                payment.save()
                
                # Create a new payment record for the refund
                refund_payment = Payment.objects.create(
                    order=payment.order,
                    payment_method=payment.payment_method,
                    amount=-refund_amount,  # Negative amount for refund
                    currency=payment.currency,
                    status='completed',
                    transaction_id=result.get('transaction_id'),
                    mobile_money_phone=payment.mobile_money_phone,
                    mobile_money_provider=payment.mobile_money_provider
                )
                
                result['refund_payment_id'] = refund_payment.id
            
            return result
            
        except Exception as e:
            logger.error(f"Payment refund error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_available_payment_methods(self, country='TZ', currency='TZS', amount=None):
        """Get available payment methods for a given country and currency"""
        available_methods = []
        
        # Get all available gateways
        gateways = self.factory.get_available_gateways()
        
        for gateway_info in gateways:
            gateway = gateway_info['instance']
            
            # Check if gateway supports the country and currency
            supported_countries = gateway.get_supported_countries()
            supported_currencies = gateway.get_supported_currencies()
            
            if (not supported_countries or country in supported_countries) and \
               (not supported_currencies or currency in supported_currencies):
                
                # Calculate fees if amount is provided
                fees = None
                if amount:
                    fees = gateway.calculate_fee(amount)
                
                available_methods.append({
                    'gateway': gateway_info['gateway'],
                    'display_name': gateway_info['display_name'],
                    'fees': fees,
                    'supports_refund': hasattr(gateway, 'refund_payment'),
                    'environment': gateway.environment
                })
        
        return available_methods

# ==================== SIMPLE FALLBACK PAYMENT SERVICE ====================

class SimplePaymentService:
    """Simple fallback payment service for testing"""
    
    def __init__(self):
        self.payments = {}
    
    def initiate_payment(self, payment):
        """Simulate payment initiation"""
        import uuid
        import time
        
        transaction_id = f"SIM-{uuid.uuid4().hex[:12].upper()}"
        
        # Store payment reference
        self.payments[transaction_id] = {
            'payment_id': payment.id,
            'amount': float(payment.amount),
            'currency': payment.currency,
            'status': 'pending',
            'created_at': time.time()
        }
        
        # Update payment record
        payment.transaction_id = transaction_id
        payment.status = 'processing'
        payment.save()
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'message': 'Payment initiated (simulation)',
            'instruction': 'This is a simulated payment. Use admin interface to complete.'
        }
    
    def verify_payment(self, payment):
        """Simulate payment verification"""
        if not payment.transaction_id:
            return {
                'success': False,
                'error': 'No transaction ID'
            }
        
        payment_data = self.payments.get(payment.transaction_id)
        
        if not payment_data:
            return {
                'success': False,
                'error': 'Payment not found'
            }
        
        # For simulation, we'll keep it pending
        return {
            'success': True,
            'status': 'pending',
            'transaction_id': payment.transaction_id,
            'message': 'Payment verification (simulation)'
        }
    
    def complete_payment(self, transaction_id):
        """Manually complete a simulated payment"""
        payment_data = self.payments.get(transaction_id)
        
        if not payment_data:
            return {
                'success': False,
                'error': 'Payment not found'
            }
        
        payment_data['status'] = 'completed'
        
        # Update actual payment record
        try:
            payment = Payment.objects.get(id=payment_data['payment_id'])
            payment.status = 'completed'
            payment.paid_at = timezone.now()
            payment.save()
            
            # Update order status
            payment.order.status = 'confirmed'
            payment.order.save()
            
            return {
                'success': True,
                'message': 'Payment completed successfully'
            }
            
        except Payment.DoesNotExist:
            return {
                'success': False,
                'error': 'Payment record not found'
            }
    
    def cancel_payment(self, transaction_id):
        """Manually cancel a simulated payment"""
        payment_data = self.payments.get(transaction_id)
        
        if not payment_data:
            return {
                'success': False,
                'error': 'Payment not found'
            }
        
        payment_data['status'] = 'cancelled'
        
        # Update actual payment record
        try:
            payment = Payment.objects.get(id=payment_data['payment_id'])
            payment.status = 'cancelled'
            payment.save()
            
            return {
                'success': True,
                'message': 'Payment cancelled'
            }
            
        except Payment.DoesNotExist:
            return {
                'success': False,
                'error': 'Payment record not found'
            }

# ==================== INSTANCE CREATION ====================

# Create global instances
try:
    payment_service = PaymentService()
    PAYMENT_SERVICES_AVAILABLE = True
    logger.info("Payment services loaded successfully")
except Exception as e:
    logger.warning(f"Payment services not available: {e}")
    payment_service = SimplePaymentService()
    PAYMENT_SERVICES_AVAILABLE = False