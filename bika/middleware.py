from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin
import time

class SecurityHeadersMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        # Add security headers to all responses
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        
        # Only set no-cache for authenticated users on sensitive pages
        if request.user.is_authenticated:
            sensitive_paths = ['/dashboard/', '/profile/', '/settings/', '/vendor/', '/admin/']
            if any(request.path.startswith(path) for path in sensitive_paths):
                response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response['Pragma'] = 'no-cache'
        
        return response

class SessionTimeoutMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if request.user.is_authenticated:
            # Update last activity timestamp
            request.session['last_activity'] = str(time.time())
        
        return None

class RoleBasedAccessMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # Skip for static files, admin, and public pages
        public_paths = ['/static/', '/admin/', '/login/', '/logout/', '/register/', '/', '/about/', '/contact/', '/faq/']
        if any(request.path.startswith(path) for path in public_paths):
            return None
        
        # Define restricted URLs for each role
        restricted_urls = {
            'customer': [
                '/admin/', '/vendor/dashboard/', '/storage-sites/', '/dashboard/',
                '/manage-datasets/', '/track-products/', '/vendor/products/'
            ],
            'vendor': [
                '/admin/', '/storage-sites/', '/manage-datasets/'
            ]
        }
        
        if request.user.is_authenticated:
            user_role = request.user.user_type
            
            # Check if current path is restricted for this user role
            for restricted_url in restricted_urls.get(user_role, []):
                if request.path.startswith(restricted_url):
                    from django.contrib import messages
                    messages.error(request, "Access denied. You don't have permission to view this page.")
                    return redirect('bika:home')
        
        return None