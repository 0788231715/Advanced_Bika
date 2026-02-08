# bika/middleware.py
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin
import time
from django.contrib import messages

class SecurityHeadersMiddleware(MiddlewareMixin):
    """
    Middleware to add security headers to all responses
    """
    def process_response(self, request, response):
        # Add security headers to all responses
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        
        # CSP Header (Content Security Policy)
        csp = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
            "style-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
        )
        response['Content-Security-Policy'] = csp
        
        # Referrer Policy
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy
        response['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        # Only set no-cache for authenticated users on sensitive pages
        if request.user.is_authenticated:
            sensitive_paths = [
                '/dashboard/', '/profile/', '/settings/', '/vendor/', '/admin/',
                '/account/', '/payment/', '/orders/', '/inventory/', '/storage/'
            ]
            if any(request.path.startswith(path) for path in sensitive_paths):
                response['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
                response['Pragma'] = 'no-cache'
                response['Expires'] = '0'
            else:
                # Cache public pages for 1 hour
                response['Cache-Control'] = 'public, max-age=3600'
        else:
            # Cache public pages for 1 hour
            response['Cache-Control'] = 'public, max-age=3600'
        
        return response

class SessionTimeoutMiddleware(MiddlewareMixin):
    """
    Middleware to handle session timeout and automatic logout
    """
    def __init__(self, get_response):
        super().__init__(get_response)
        self.session_timeout = 3600  # 1 hour in seconds
        self.warning_time = 300      # 5 minutes warning
    
    def process_request(self, request):
        if request.user.is_authenticated:
            current_time = time.time()
            
            # Check if session has last_activity timestamp
            if 'last_activity' in request.session:
                last_activity = float(request.session['last_activity'])
                time_since_last_activity = current_time - last_activity
                
                # Check if session has expired
                if time_since_last_activity > self.session_timeout:
                    # Session expired, log out user
                    from django.contrib.auth import logout
                    logout(request)
                    messages.warning(request, "Your session has expired. Please log in again.")
                    
                    # Store the original path to redirect back after login
                    if not any(request.path.startswith(path) for path in ['/login/', '/logout/', '/register/']):
                        request.session['next'] = request.get_full_path()
                    
                    return redirect('bika:login')
                
                # Check if session is about to expire (5 minutes warning)
                elif time_since_last_activity > (self.session_timeout - self.warning_time):
                    # Add warning to session if not already there
                    if 'session_warning_shown' not in request.session:
                        request.session['session_warning_shown'] = True
                        messages.warning(
                            request, 
                            f"Your session will expire in {int((self.session_timeout - time_since_last_activity) // 60)} minutes. "
                            "Please save your work."
                        )
            
            # Update last activity timestamp
            request.session['last_activity'] = str(current_time)
        
        return None
    
    def process_response(self, request, response):
        # Remove session warning flag if user performed an action
        if request.user.is_authenticated and 'session_warning_shown' in request.session:
            # Check if this is a new request (not the same as when warning was shown)
            if 'last_activity' in request.session:
                current_time = time.time()
                last_activity = float(request.session['last_activity'])
                if current_time - last_activity < 60:  # If activity within last minute
                    del request.session['session_warning_shown']
        
        return response

class RoleBasedAccessMiddleware(MiddlewareMixin):
    """
    Middleware to enforce role-based access control
    """
    def process_request(self, request):
        # Skip middleware for these paths
        public_paths = [
            '/static/', '/media/', '/admin/', '/login/', '/logout/', 
            '/register/', '/password-reset/', '/api/', '/webhook/',
            '/', '/about/', '/contact/', '/faq/', '/services/', 
            '/testimonials/', '/privacy/', '/terms/', '/products/'
        ]
        
        if any(request.path.startswith(path) for path in public_paths):
            return None
        
        # Define role-based access rules
        role_access_rules = {
            'customer': {
                'allowed': [
                    '/dashboard/', '/profile/', '/settings/', '/orders/', 
                    '/wishlist/', '/cart/', '/checkout/', '/payment/',
                    '/notifications/', '/reviews/', '/track-order/',
                    '/api/user/', '/api/orders/', '/api/cart/'
                ],
                'restricted': [
                    '/admin/', '/vendor/', '/storage/', '/inventory/',
                    '/deliveries/', '/reports/', '/analytics/', '/manage/',
                    '/api/vendor/', '/api/storage/', '/api/inventory/',
                    '/api/deliveries/', '/api/analytics/'
                ]
            },
            'vendor': {
                'allowed': [
                    '/vendor/dashboard/', '/vendor/products/', '/vendor/orders/',
                    '/vendor/settings/', '/vendor/analytics/', '/vendor/reports/',
                    '/api/vendor/products/', '/api/vendor/orders/',
                    '/dashboard/', '/profile/', '/settings/', '/notifications/',
                    '/track-order/', '/api/user/', '/api/vendor/'
                ],
                'restricted': [
                    '/admin/', '/storage/', '/inventory/', '/deliveries/',
                    '/manage-datasets/', '/ai-models/', '/system-settings/',
                    '/api/admin/', '/api/storage/', '/api/inventory/',
                    '/api/system/', '/api/ai-models/'
                ]
            },
            'admin': {
                'allowed': [
                    '/admin/', '/dashboard/', '/profile/', '/settings/',
                    '/reports/', '/analytics/', '/storage/', '/inventory/',
                    '/deliveries/', '/manage-datasets/', '/ai-models/',
                    '/system-settings/', '/api/', '/webhook/'
                ],
                'restricted': []
            },
            'storage_staff': {
                'allowed': [
                    '/storage/dashboard/', '/storage/inventory/', 
                    '/storage/tracking/', '/storage/reports/',
                    '/profile/', '/settings/', '/notifications/',
                    '/api/storage/', '/api/inventory/'
                ],
                'restricted': [
                    '/admin/', '/vendor/', '/manage-datasets/', '/ai-models/',
                    '/system-settings/', '/api/admin/', '/api/vendor/',
                    '/api/system/', '/api/ai-models/'
                ]
            },
            'negotiation_team': {
                'allowed': [
                    '/negotiation/dashboard/', '/negotiation/clients/',
                    '/negotiation/contracts/', '/negotiation/reports/',
                    '/profile/', '/settings/', '/notifications/',
                    '/api/negotiation/'
                ],
                'restricted': [
                    '/admin/', '/vendor/', '/storage/', '/inventory/',
                    '/deliveries/', '/manage-datasets/', '/ai-models/',
                    '/system-settings/', '/api/admin/', '/api/vendor/',
                    '/api/storage/', '/api/inventory/', '/api/deliveries/',
                    '/api/system/', '/api/ai-models/'
                ]
            }
        }
        
        if request.user.is_authenticated:
            # Get user role from UserRole model if exists, otherwise use user_type
            try:
                user_role_obj = request.user.user_role
                user_role = user_role_obj.role
            except:
                # Fallback to user_type for backward compatibility
                user_role = request.user.user_type
            
            # Check if user has appropriate permissions
            if hasattr(request.user, 'user_role'):
                permissions = request.user.user_role.permissions
                if isinstance(permissions, dict):
                    # Check custom permissions if they exist
                    if 'allowed_urls' in permissions:
                        allowed_urls = permissions['allowed_urls']
                        if request.path not in allowed_urls and not any(
                            request.path.startswith(url) for url in allowed_urls
                        ):
                            messages.error(request, "Access denied. You don't have permission to access this page.")
                            return redirect('bika:dashboard')
            
            # Get access rules for user's role
            access_rules = role_access_rules.get(user_role, {})
            allowed_paths = access_rules.get('allowed', [])
            restricted_paths = access_rules.get('restricted', [])
            
            # Check if current path is explicitly restricted
            for restricted_path in restricted_paths:
                if request.path.startswith(restricted_path):
                    messages.error(
                        request, 
                        f"Access denied. {user_role.replace('_', ' ').title()}s cannot access this page."
                    )
                    return redirect('bika:dashboard')
            
            # For non-admin users, check if path is in allowed list
            if user_role != 'admin':
                # Allow access to allowed paths
                is_allowed = False
                for allowed_path in allowed_paths:
                    if request.path.startswith(allowed_path):
                        is_allowed = True
                        break
                
                # If not in allowed list and not a public path, deny access
                if not is_allowed and not any(
                    request.path.startswith(path) for path in public_paths
                ):
                    messages.error(
                        request, 
                        "Access denied. You don't have permission to view this page."
                    )
                    return redirect('bika:dashboard')
        
        else:
            # For unauthenticated users trying to access protected areas
            protected_paths = [
                '/dashboard/', '/profile/', '/settings/', '/vendor/',
                '/storage/', '/inventory/', '/deliveries/', '/negotiation/',
                '/orders/', '/cart/', '/checkout/', '/payment/', '/api/user/',
                '/api/orders/', '/api/cart/', '/api/vendor/', '/api/storage/',
                '/api/inventory/', '/api/deliveries/', '/api/negotiation/'
            ]
            
            if any(request.path.startswith(path) for path in protected_paths):
                messages.warning(request, "Please log in to access this page.")
                request.session['next'] = request.get_full_path()
                return redirect('bika:login')
        
        return None
    
    def process_exception(self, request, exception):
        """Handle exceptions in role-based access"""
        # Log access violations
        import logging
        logger = logging.getLogger('bika.security')
        
        if request.user.is_authenticated:
            logger.warning(
                f"Access violation attempt by user {request.user.username} "
                f"({request.user.user_type}) at {request.path}"
            )
        else:
            logger.warning(
                f"Unauthorized access attempt at {request.path} from IP: {request.META.get('REMOTE_ADDR')}"
            )
        
        return None