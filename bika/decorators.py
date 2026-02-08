from django.http import HttpResponseForbidden
from django.shortcuts import redirect
from functools import wraps

def role_required(allowed_roles=[]):
    """
    Decorator to check if user has required role.
    Usage: @role_required(['admin', 'staff'])
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            
            # Check if user has any of the allowed roles
            # Adjust this logic based on your user model
            user_role = getattr(request.user, 'role', None)
            
            if user_role in allowed_roles:
                return view_func(request, *args, **kwargs)
            
            # You can also check groups
            user_groups = request.user.groups.all()
            group_names = [group.name for group in user_groups]
            
            for group in group_names:
                if group in allowed_roles:
                    return view_func(request, *args, **kwargs)
            
            return HttpResponseForbidden("You don't have permission to access this page.")
        return _wrapped_view
    return decorator


def admin_required(view_func):
    """Decorator for admin-only access"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        # Check if user is admin or superuser
        if hasattr(request.user, 'is_admin') and request.user.is_admin:
            return view_func(request, *args, **kwargs)
        
        if request.user.is_superuser:
            return view_func(request, *args, **kwargs)
        
        # Check admin group
        if request.user.groups.filter(name='admin').exists():
            return view_func(request, *args, **kwargs)
        
        return HttpResponseForbidden("Admin access required.")
    return _wrapped_view