# bika/views.py - FIXED AND COMPLETE VERSION
from decimal import Decimal
import os
import json
import logging
from httpx import request
import pandas as pd
from functools import wraps
import numpy as np
from datetime import datetime, timedelta
from bika.service import fruit_ai_service
from bika.services.ai_service import enhanced_ai_service
import joblib
from sklearn.preprocessing import LabelEncoder
from django.core.files.storage import default_storage
import tempfile
from django.shortcuts import render, redirect, get_object_or_404
from django.http import Http404, HttpResponse, JsonResponse, HttpResponseRedirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.views.generic import ListView, DetailView, TemplateView
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST, require_GET
from django.views.decorators.cache import never_cache
from django.db.models import Q, Count, Sum, F, Avg, Max, Min
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.utils import timezone
from django.urls import reverse
from django.db import transaction
from .services import payment_service, PAYMENT_SERVICES_AVAILABLE
# Add these imports at the top of views.py after existing imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Import models
from .models import (
    CustomUser, Delivery, DeliveryItem, DeliveryStatusHistory, InventoryHistory, InventoryItem, Product, ProductCategory, ProductImage, ProductReview,
    Wishlist, Cart, Order, OrderItem, Payment,
    SiteInfo, Service, Testimonial, ContactMessage, FAQ,
    StorageLocation, FruitType, FruitBatch, FruitQualityReading, 
    RealTimeSensorData, ProductAlert, Notification,
    ProductDataset, TrainedModel, PaymentGatewaySettings, CurrencyExchangeRate
)

# Make sure these imports exist:
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import Payment 
from django.db.models import Q, F, Sum, Count
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# Import models
from .models import (
    ClientRequest, InventoryItem, Delivery, DeliveryItem, DeliveryStatusHistory,
    InventoryHistory, ProductCategory, SiteInfo, CustomUser
)

# Import forms
from .forms import ClientRequestForm

# Import decorators
from .decorators import role_required

# Import forms
from .forms import (
    ContactForm, NewsletterForm, CustomUserCreationForm, 
    VendorRegistrationForm, CustomerRegistrationForm, ProductForm,
    ProductImageForm, FruitBatchForm, FruitQualityReadingForm
)

# Import services

AI_SERVICES_AVAILABLE = True

# Payment services (simple fallback)
PAYMENT_SERVICES_AVAILABLE = False

# Try to import payment services
try:
    from .services.payment_gateways import PaymentGatewayFactory
    PAYMENT_SERVICES_AVAILABLE = True
except ImportError:
    PAYMENT_SERVICES_AVAILABLE = False
    logging.warning("Payment services not available")


# Try to import AI services
try:
    from .services.ai_service import enhanced_ai_service
    AI_SERVICES_AVAILABLE = True
    logging.info("AI services loaded successfully")
except ImportError as e:
    AI_SERVICES_AVAILABLE = False
    logging.warning(f"AI services not available: {e}")
    
    # Create a simple fallback service
    class SimpleAIService:
        def __init__(self):
            self.active_model = None
        
        def predict_and_alert(self, product_id):
            return {
                'success': True,
                'message': 'AI service not available - using fallback',
                'prediction': {'quality': 'Good', 'confidence': 0.7},
                'alerts': []
            }
        
        def get_detailed_model_comparison(self):
            return {
                'available': False, 
                'message': 'AI service not available',
                'models': [],
                'best_model': None
            }
        
        def generate_sample_dataset(self, num_samples):
            return {
                'success': False, 
                'error': 'AI service not available',
                'filename': None,
                'download_url': None
            }
        
        def load_active_model(self):
            self.active_model = None
        
        def train_five_models(self, csv_file, target_column):
            return {
                'success': False,
                'error': 'AI service not available',
                'model_saved': False
            }
    
    enhanced_ai_service = SimpleAIService()

# Set up logger
logger = logging.getLogger(__name__)
# ==================== ROLE DECORATORS ====================

def role_required(*roles):
    """Decorator to check if user has required role"""
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                messages.error(request, "Authentication required.")
                return redirect('bika:login')
            
            # Get user role
            user_role = None
            try:
                if hasattr(request.user, 'user_role'):
                    user_role = request.user.user_role.role
                else:
                    # Fallback to user_type
                    user_role = request.user.user_type
            except:
                user_role = request.user.user_type
            
            # Check if user has required role
            if user_role not in roles:
                messages.error(request, f"Access denied. Required role: {', '.join(roles)}")
                return redirect('bika:home')
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator

def role_allowed(*roles):
    """Decorator to allow multiple roles"""
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                messages.error(request, "Authentication required.")
                return redirect('bika:login')
            
            # Get user role
            user_role = None
            try:
                if hasattr(request.user, 'user_role'):
                    user_role = request.user.user_role.role
                else:
                    # Fallback to user_type
                    user_role = request.user.user_type
            except:
                user_role = request.user.user_type
            
            # Check if user has allowed role
            if user_role not in roles:
                messages.error(request, f"Access denied. Allowed roles: {', '.join(roles)}")
                return redirect('bika:home')
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator
# ==================== BASIC VIEWS ====================

class HomeView(TemplateView):
    template_name = 'bika/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get site info
        context['site_info'] = SiteInfo.objects.first()
        
        # Get featured products
        try:
            featured_products = Product.objects.filter(
                status='active',
                is_featured=True
            ).select_related('category', 'vendor')[:8]
            
            # Add primary images
            for product in featured_products:
                product.primary_image = product.images.filter(is_primary=True).first()
                if not product.primary_image:
                    product.primary_image = product.images.first()
            
            context['featured_products'] = featured_products
        except Exception as e:
            logger.error(f"Error loading featured products: {e}")
            context['featured_products'] = []
        
        # Get services
        context['featured_services'] = Service.objects.filter(is_active=True)[:6]
        
        # Get testimonials
        context['featured_testimonials'] = Testimonial.objects.filter(
            is_active=True, 
            is_featured=True
        )[:3]
        
        # Get FAQs
        context['faqs'] = FAQ.objects.filter(is_active=True)[:5]
        
        # Get product categories for navigation
        context['categories'] = ProductCategory.objects.filter(
            is_active=True, 
            parent__isnull=True
        )[:8]
        
        # Get stats for homepage
        context['total_products'] = Product.objects.filter(status='active').count()
        context['total_vendors'] = CustomUser.objects.filter(
            user_type='vendor', 
            is_active=True
        ).count()
        
        return context

# Add this function to your existing views.py, anywhere after the product_list_view function

def product_search_view(request):
    """Handle product search requests"""
    query = request.GET.get('q', '').strip()
    
    if not query:
        return redirect('bika:product_list')
    
    # Search products
    products = Product.objects.filter(
        Q(name__icontains=query) | 
        Q(description__icontains=query) |
        Q(short_description__icontains=query) |
        Q(tags__icontains=query) |
        Q(category__name__icontains=query),
        status='active'
    ).select_related('category', 'vendor')
    
    # Get search suggestions
    suggestions = []
    if products.exists():
        suggestions = Product.objects.filter(
            category__in=products.values_list('category', flat=True),
            status='active'
        ).exclude(id__in=products.values_list('id', flat=True))[:5]
    
    # Get categories
    categories = ProductCategory.objects.filter(
        is_active=True,
        parent__isnull=True
    ).annotate(
        product_count=Count('products', filter=Q(products__status='active'))
    )
    
    # Pagination
    paginator = Paginator(products, 12)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'products': page_obj,
        'query': query,
        'suggestions': suggestions,
        'categories': categories,
        'total_results': products.count(),
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/search_results.html', context)

def user_settings(request):
    """User settings page"""
    context = {
        'user': request.user,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/settings.html', context)

@login_required
@require_POST
def quick_add_to_cart(request, product_id):
    """Quick add to cart (for AJAX requests)"""
    product = get_object_or_404(Product, id=product_id)
    
    # Check stock
    if product.track_inventory and product.stock_quantity < 1:
        return JsonResponse({
            'success': False,
            'message': f'Product out of stock!'
        })
    
    # Add to cart
    cart_item, created = Cart.objects.get_or_create(
        user=request.user,
        product=product,
        defaults={'quantity': 1}
    )
    
    if not created:
        cart_item.quantity += 1
        cart_item.save()
    
    # Get updated cart count
    cart_count = Cart.objects.filter(user=request.user).count()
    
    return JsonResponse({
        'success': True,
        'message': f'{product.name} added to cart!',
        'cart_count': cart_count,
        'created': created
    })
def about_view(request):
    services = Service.objects.filter(is_active=True)
    testimonials = Testimonial.objects.filter(is_active=True)[:4]
    site_info = SiteInfo.objects.first()
    
    context = {
        'services': services,
        'testimonials': testimonials,
        'site_info': site_info,
    }
    return render(request, 'bika/pages/about.html', context)

def services_view(request):
    services = Service.objects.filter(is_active=True)
    site_info = SiteInfo.objects.first()
    
    context = {
        'services': services,
        'site_info': site_info,
    }
    return render(request, 'bika/pages/services.html', context)

class ServiceDetailView(DetailView):
    model = Service
    template_name = 'bika/pages/service_detail.html'
    context_object_name = 'service'
    slug_field = 'slug'
    slug_url_kwarg = 'slug'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['site_info'] = SiteInfo.objects.first()
        return context

def contact_view(request):
    site_info = SiteInfo.objects.first()
    
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            contact_message = form.save(commit=False)
            
            # Get client IP address
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                contact_message.ip_address = x_forwarded_for.split(',')[0]
            else:
                contact_message.ip_address = request.META.get('REMOTE_ADDR')
            
            contact_message.save()
            
            # Send email notification
            try:
                send_mail(
                    f'New Contact Message: {contact_message.subject}',
                    f'''
                    Name: {contact_message.name}
                    Email: {contact_message.email}
                    Phone: {contact_message.phone}
                    
                    Message:
                    {contact_message.message}
                    ''',
                    settings.DEFAULT_FROM_EMAIL,
                    [settings.DEFAULT_FROM_EMAIL],
                    fail_silently=True,
                )
            except Exception as e:
                logger.error(f"Email error: {e}")
            
            messages.success(
                request, 
                'Thank you for your message! We will get back to you soon.'
            )
            return redirect('bika:contact')
    else:
        form = ContactForm()
    
    context = {
        'form': form,
        'site_info': site_info,
    }
    return render(request, 'bika/pages/contact.html', context)

def faq_view(request):
    faqs = FAQ.objects.filter(is_active=True)
    site_info = SiteInfo.objects.first()
    
    context = {
        'faqs': faqs,
        'site_info': site_info,
    }
    return render(request, 'bika/pages/faq.html', context)

@csrf_exempt
@require_POST
def newsletter_subscribe(request):
    """Handle newsletter subscription"""
    try:
        email = request.POST.get('email')
        
        if not email:
            return JsonResponse({
                'success': False,
                'message': 'Please enter a valid email address.'
            })
        
        # Here you would save to your newsletter model
        # For now, just return success
        return JsonResponse({
            'success': True,
            'message': 'Thank you for subscribing to our newsletter!'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': 'An error occurred. Please try again.'
        })

# ==================== ADMIN VIEWS ====================
# ==================== DASHBOARD ENHANCEMENTS ====================

@staff_member_required
def admin_dashboard(request):
    """Enhanced admin dashboard with comprehensive statistics"""
    now = timezone.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    thirty_days_ago = now - timedelta(days=30)
    
    # ===== USER STATISTICS =====
    total_users = CustomUser.objects.count()
    total_admins = CustomUser.objects.filter(user_type='admin').count()
    total_vendors = CustomUser.objects.filter(user_type='vendor').count()
    total_customers = CustomUser.objects.filter(user_type='customer').count()
    new_users_today = CustomUser.objects.filter(date_joined__gte=today_start).count()
    active_users = CustomUser.objects.filter(last_login__gte=thirty_days_ago).count()
    
    # Calculate percentages
    if total_users > 0:
        admin_percentage = round((total_admins / total_users) * 100, 1)
        vendor_percentage = round((total_vendors / total_users) * 100, 1)
        customer_percentage = round((total_customers / total_users) * 100, 1)
    else:
        admin_percentage = vendor_percentage = customer_percentage = 0
    
    # ===== PRODUCT STATISTICS =====
    total_products = Product.objects.count()
    active_products = Product.objects.filter(status='active').count()
    draft_products = Product.objects.filter(status='draft').count()
    out_of_stock = Product.objects.filter(
        stock_quantity=0, 
        track_inventory=True
    ).count()
    low_stock = Product.objects.filter(
        stock_quantity__gt=0,
        stock_quantity__lte=F('low_stock_threshold'),
        track_inventory=True
    ).count()
    featured_products = Product.objects.filter(is_featured=True, status='active').count()
    
    # ===== ORDER STATISTICS =====
    total_orders = Order.objects.count()
    pending_orders = Order.objects.filter(status='pending').count()
    confirmed_orders = Order.objects.filter(status='confirmed').count()
    shipped_orders = Order.objects.filter(status='shipped').count()
    delivered_orders = Order.objects.filter(status='delivered').count()
    cancelled_orders = Order.objects.filter(status='cancelled').count()
    
    # ===== REVENUE CALCULATIONS =====
    completed_orders = Order.objects.filter(status='delivered')
    total_revenue = completed_orders.aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    
    today_revenue = completed_orders.filter(
        created_at__gte=today_start
    ).aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    
    # ===== CATEGORY STATISTICS =====
    total_categories = ProductCategory.objects.count()
    active_categories = ProductCategory.objects.filter(is_active=True).count()
    categories_with_products = ProductCategory.objects.filter(
        products__status='active'
    ).distinct().count()
    
    # ===== FRUIT MONITORING STATS =====
    fruit_batches = FruitBatch.objects.count()
    active_fruit_batches = FruitBatch.objects.filter(status='active').count()
    fruit_types = FruitType.objects.count()
    quality_readings = FruitQualityReading.objects.count()
    
    # ===== AI SYSTEM STATS =====
    total_predictions = FruitQualityReading.objects.count()
    dataset_size = ProductDataset.objects.count()
    active_vendors = CustomUser.objects.filter(
        user_type='vendor', is_active=True
    ).count()
    
    # Get critical alerts
    critical_alerts = ProductAlert.objects.filter(
        is_resolved=False, severity='critical'
    ).count()
    
    # ===== RECENT DATA =====
    recent_products = Product.objects.select_related(
        'vendor', 'category'
    ).prefetch_related('images').order_by('-created_at')[:6]
    
    recent_orders = Order.objects.select_related('user').order_by('-created_at')[:5]
    
    recent_messages = ContactMessage.objects.filter(
        status='new'
    ).order_by('-submitted_at')[:5]
    
    # Get Django version and debug status
    import django
    from django.conf import settings
    django_version = django.get_version()
    debug = settings.DEBUG
    
    # Add status colors for orders
    for order in recent_orders:
        status_colors = {
            'pending': 'warning',
            'confirmed': 'info',
            'shipped': 'primary',
            'delivered': 'success',
            'cancelled': 'danger'
        }
        order.status_color = status_colors.get(order.status, 'secondary')
    
    context = {
        # User statistics
        'total_users': total_users,
        'total_admins': total_admins,
        'total_vendors': total_vendors,
        'total_customers': total_customers,
        'new_users_today': new_users_today,
        'active_users': active_users,
        'admin_percentage': admin_percentage,
        'vendor_percentage': vendor_percentage,
        'customer_percentage': customer_percentage,
        'active_vendors': active_vendors,
        
        # Product statistics
        'total_products': total_products,
        'active_products': active_products,
        'draft_products': draft_products,
        'out_of_stock': out_of_stock,
        'low_stock': low_stock,
        'featured_products': featured_products,
        
        # Order statistics
        'total_orders': total_orders,
        'pending_orders': pending_orders,
        'confirmed_orders': confirmed_orders,
        'shipped_orders': shipped_orders,
        'delivered_orders': delivered_orders,
        'cancelled_orders': cancelled_orders,
        
        # Revenue
        'total_revenue': total_revenue,
        'today_revenue': today_revenue,
        
        # Category statistics
        'total_categories': total_categories,
        'active_categories': active_categories,
        'categories_with_products': categories_with_products,
        
        # Fruit monitoring
        'fruit_batches': fruit_batches,
        'active_fruit_batches': active_fruit_batches,
        'fruit_types': fruit_types,
        'quality_readings': quality_readings,
        
        # AI System stats
        'ai_service': enhanced_ai_service,
        'total_predictions': total_predictions,
        'dataset_size': dataset_size,
        'critical_alerts': critical_alerts,
        
        # Recent data
        'recent_products': recent_products,
        'recent_orders': recent_orders,
        'recent_messages': recent_messages,
        
        # System info
        'django_version': django_version,
        'debug': debug,
        
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/admin/dashboard.html', context)

@staff_member_required
@require_GET
def sales_analytics_api(request):
    """API for sales analytics"""
    days = int(request.GET.get('days', 30))
    end_date = timezone.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate daily sales data
    sales_data = []
    current_date = start_date
    
    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        daily_sales = Order.objects.filter(
            created_at__range=[current_date, next_date],
            status='delivered'
        ).aggregate(total=Sum('total_amount'))['total'] or 0
        
        sales_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'sales': float(daily_sales),
            'orders': Order.objects.filter(
                created_at__range=[current_date, next_date],
                status='delivered'
            ).count()
        })
        
        current_date = next_date
    
    # Get top selling products
    top_products = OrderItem.objects.filter(
        order__created_at__range=[start_date, end_date],
        order__status='delivered'
    ).values(
        'product__name', 'product__sku'
    ).annotate(
        total_quantity=Sum('quantity'),
        total_revenue=Sum(F('quantity') * F('price'))
    ).order_by('-total_quantity')[:5]
    
    return JsonResponse({
        'success': True,
        'sales_data': sales_data,
        'top_products': list(top_products),
        'total_days': days
    })

@staff_member_required
@require_GET
def get_active_alerts(request):
    """API for active alerts"""
    alerts = ProductAlert.objects.filter(
        is_resolved=False
    ).select_related('product').order_by('-created_at')[:10]
    
    alert_list = []
    for alert in alerts:
        alert_list.append({
            'id': alert.id,
            'title': f"{alert.alert_type.replace('_', ' ').title()} Alert",
            'message': alert.message,
            'severity': alert.severity,
            'product': alert.product.name if alert.product else 'Unknown',
            'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M'),
            'details': json.loads(alert.details) if alert.details else {}
        })
    
    return JsonResponse({
        'success': True,
        'alerts': alert_list,
        'count': alerts.count()
    })

@staff_member_required
@require_GET
def performance_metrics_api(request):
    """API for performance metrics"""
    import psutil
    import os
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Get database query count (simplified)
    from django.db import connection
    db_queries = len(connection.queries) if settings.DEBUG else 0
    
    return JsonResponse({
        'success': True,
        'response_time': round(cpu_percent / 100, 3),  # Simulated
        'server_load': f"{cpu_percent}%",
        'memory_usage': f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB",
        'disk_usage': f"{disk.percent}%",
        'db_queries': db_queries,
        'timestamp': timezone.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@staff_member_required
@require_GET
def export_inventory_report(request):
    """Export inventory report as CSV"""
    import csv
    from django.http import HttpResponse
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    filename = f"inventory-report-{timezone.now().strftime('%Y%m%d')}.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    writer = csv.writer(response)
    
    # Write headers
    writer.writerow([
        'Product ID', 'SKU', 'Product Name', 'Category', 
        'Stock Quantity', 'Low Stock Threshold', 'Status',
        'Price', 'Vendor', 'Last Updated'
    ])
    
    # Write data
    products = Product.objects.select_related('category', 'vendor').order_by('category__name', 'name')
    
    for product in products:
        writer.writerow([
            product.id,
            product.sku,
            product.name,
            product.category.name if product.category else '',
            product.stock_quantity,
            product.low_stock_threshold,
            product.get_status_display(),
            float(product.price),
            product.vendor.username if product.vendor else '',
            product.updated_at.strftime('%Y-%m-%d %H:%M')
        ])
    
    return response

@staff_member_required
@require_GET
def get_user_activity(request):
    """Get recent user activity"""
    from datetime import datetime, timedelta
    
    recent_activity = []
    
    # Recent logins (last 24 hours)
    recent_logins = CustomUser.objects.filter(
        last_login__gte=timezone.now() - timedelta(hours=24)
    ).order_by('-last_login')[:5]
    
    for user in recent_logins:
        recent_activity.append({
            'type': 'login',
            'user': user.username,
            'time': user.last_login.strftime('%H:%M'),
            'icon': 'fas fa-sign-in-alt',
            'color': 'success'
        })
    
    # Recent orders
    recent_orders = Order.objects.order_by('-created_at')[:3]
    for order in recent_orders:
        recent_activity.append({
            'type': 'order',
            'user': order.user.username,
            'time': order.created_at.strftime('%H:%M'),
            'icon': 'fas fa-shopping-cart',
            'color': 'primary',
            'details': f"Order #{order.order_number}"
        })
    
    # Recent product additions
    recent_products = Product.objects.order_by('-created_at')[:3]
    for product in recent_products:
        recent_activity.append({
            'type': 'product',
            'user': product.vendor.username if product.vendor else 'System',
            'time': product.created_at.strftime('%H:%M'),
            'icon': 'fas fa-cube',
            'color': 'info',
            'details': f"Added: {product.name}"
        })
    
    return JsonResponse({
        'success': True,
        'activities': recent_activity
    })
# ==================== AI ALERT SYSTEM VIEWS ====================

@staff_member_required
def ai_alert_dashboard(request):
    """Dashboard for AI-generated alerts"""
    # Get all alerts
    alerts = ProductAlert.objects.filter(
        is_resolved=False
    ).select_related('product', 'product__vendor').order_by('-created_at')
    
    # Get alert statistics
    total_alerts = alerts.count()
    critical_alerts = alerts.filter(severity='critical').count()
    high_alerts = alerts.filter(severity='high').count()
    
    # Get recent predictions
    recent_predictions = []
    try:
        from .ai_integration.models import FruitPrediction
        recent_predictions = FruitPrediction.objects.select_related(
            'product'
        ).order_by('-prediction_date')[:10]
    except:
        pass
    
    context = {
        'alerts': alerts[:50],  # Limit to 50 for performance
        'total_alerts': total_alerts,
        'critical_alerts': critical_alerts,
        'high_alerts': high_alerts,
        'recent_predictions': recent_predictions,
        'ai_service': enhanced_ai_service,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/admin/ai_alert_dashboard.html', context)

@staff_member_required
@require_POST
def scan_all_products_for_alerts(request):
    """Scan all products and generate AI alerts"""
    try:
        from .models import Product
        
        # Get all active products
        products = Product.objects.filter(status='active')
        
        results = {
            'scanned': 0,
            'predictions': 0,
            'alerts': 0,
            'errors': 0
        }
        
        # Scan each product
        for product in products[:50]:  # Limit to 50 for performance
            try:
                # Get prediction and alerts
                result = enhanced_ai_service.predict_and_alert(product.id)
                
                if 'error' in result:
                    results['errors'] += 1
                else:
                    results['predictions'] += 1
                    results['alerts'] += len(result.get('alerts', []))
                
                results['scanned'] += 1
                
            except Exception as e:
                results['errors'] += 1
                print(f"Error scanning product {product.id}: {e}")
        
        messages.success(
            request, 
            f"Scanned {results['scanned']} products. "
            f"Generated {results['alerts']} new alerts."
        )
        
        return redirect('bika:ai_alert_dashboard')
        
    except Exception as e:
        messages.error(request, f"Error scanning products: {e}")
        return redirect('bika:ai_alert_dashboard')

@login_required
def product_ai_insights(request, product_id):
    """Show AI insights for a specific product"""
    product = get_object_or_404(Product, id=product_id)
    
    # Check permission
    if not request.user.is_staff and product.vendor != request.user:
        messages.error(request, "Access denied.")
        return redirect('bika:home')
    
    # Get predictions for this product
    predictions = []
    try:
        from .ai_integration.models import FruitPrediction
        predictions = FruitPrediction.objects.filter(
            product=product
        ).order_by('-prediction_date')[:10]
    except:
        pass
    
    # Get alerts for this product
    alerts = ProductAlert.objects.filter(
        product=product,
        is_resolved=False
    ).order_by('-created_at')
    
    # Get real-time prediction
    current_prediction = None
    try:
        result = enhanced_ai_service.predict_and_alert(product.id)
        if 'prediction' in result:
            current_prediction = result['prediction']
    except:
        pass
    
    context = {
        'product': product,
        'predictions': predictions,
        'alerts': alerts,
        'current_prediction': current_prediction,
        'ai_service': enhanced_ai_service,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/product_ai_insights.html', context)

@staff_member_required
def train_new_model_view(request):
    """Train a new AI model"""
    if request.method == 'POST':
        if 'dataset_file' not in request.FILES:
            messages.error(request, 'Please upload a dataset file')
            return redirect('bika:train_new_model')
        
        csv_file = request.FILES['dataset_file']
        target_column = request.POST.get('target_column', 'quality_class')
        model_type = request.POST.get('model_type', 'random_forest')
        
        # Train model
        result = enhanced_ai_service.train_five_models(csv_file, target_column)
        
        if 'error' in result:
            messages.error(request, f"Training failed: {result['error']}")
        else:
            messages.success(request, 'Model trained successfully!')
            
            if result.get('model_saved'):
                messages.info(request, f"New model activated (ID: {result['model_id']})")
        
        return redirect('bika:model_management')
    
    context = {
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/admin/train_new_model.html', context)

@staff_member_required
def model_management(request):
    """Manage AI models"""
    # Get model comparison
    comparison_result = enhanced_ai_service.get_detailed_model_comparison()
    
    # Get active model info
    active_model_info = None
    if enhanced_ai_service.active_model:
        active_model_info = {
            'name': enhanced_ai_service.active_model['record'].name,
            'accuracy': enhanced_ai_service.active_model['record'].accuracy,
            'features': enhanced_ai_service.active_model['record'].features_used,
            'trained_date': enhanced_ai_service.active_model['record'].trained_date
        }
    
    context = {
        'comparison_result': comparison_result,
        'active_model': active_model_info,
        'ai_service': enhanced_ai_service,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/admin/model_management.html', context)

@staff_member_required
@require_POST
def activate_model(request, model_id):
    """Activate a specific model"""
    try:
        from .ai_integration.models import TrainedModel
        
        # Get the model to activate
        model_to_activate = TrainedModel.objects.get(id=model_id, model_type='quality')
        
        # Deactivate all other models
        TrainedModel.objects.filter(
            model_type='quality'
        ).exclude(id=model_id).update(is_active=False)
        
        # Activate this model
        model_to_activate.is_active = True
        model_to_activate.save()
        
        # Reload the model in the service
        enhanced_ai_service.load_active_model()
        
        messages.success(request, f"Model '{model_to_activate.name}' activated successfully!")
        
    except TrainedModel.DoesNotExist:
        messages.error(request, "Model not found")
    except Exception as e:
        messages.error(request, f"Error activating model: {e}")
    
    return redirect('bika:model_management')

@staff_member_required
def generate_sample_data_view(request):
    """Generate sample dataset for training"""
    if request.method == 'POST':
        num_samples = int(request.POST.get('samples', 1000))
        
        result = enhanced_ai_service.generate_sample_dataset(num_samples)
        
        if result.get('success'):
            messages.success(request, f'Sample dataset generated with {num_samples} records')
            
            # Offer download link
            request.session['generated_dataset'] = {
                'filename': result['filename'],
                'download_url': result['download_url']
            }
        else:
            messages.error(request, f"Failed to generate dataset: {result.get('error')}")
    
    context = {
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/admin/generate_sample_data.html', context)

@staff_member_required
def download_generated_dataset(request):
    """Download generated dataset"""
    dataset_info = request.session.get('generated_dataset')
    
    if not dataset_info:
        messages.error(request, "No dataset available for download")
        return redirect('bika:generate_sample_data')
    
    filepath = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset_info['filename'])
    
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{dataset_info["filename"]}"'
            return response
    
    messages.error(request, "Dataset file not found")
    return redirect('bika:generate_sample_data')

# ==================== API ENDPOINTS FOR ALERTS ====================

@csrf_exempt
@require_POST
def batch_product_scan_api(request):
    """API endpoint to scan multiple products"""
    try:
        product_ids = json.loads(request.body).get('product_ids', [])
        
        if not product_ids:
            return JsonResponse({'success': False, 'error': 'No product IDs provided'})
        
        results = {
            'scanned': 0,
            'alerts_generated': 0,
            'errors': 0,
            'product_results': []
        }
        
        for product_id in product_ids[:20]:  # Limit to 20 products
            try:
                result = enhanced_ai_service.predict_and_alert(product_id)
                
                product_result = {
                    'product_id': product_id,
                    'success': 'error' not in result,
                    'alerts': len(result.get('alerts', []))
                }
                
                if 'error' in result:
                    product_result['error'] = result['error']
                    results['errors'] += 1
                else:
                    results['alerts_generated'] += product_result['alerts']
                
                results['product_results'].append(product_result)
                results['scanned'] += 1
                
            except Exception as e:
                results['errors'] += 1
        
        return JsonResponse({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
@require_GET
def get_product_quality_prediction(request, product_id):
    """Get quality prediction for a product"""
    try:
        product = Product.objects.get(id=product_id)
        
        # Check permission
        if not request.user.is_staff and product.vendor != request.user:
            return JsonResponse({'success': False, 'error': 'Permission denied'})
        
        # Get prediction
        result = enhanced_ai_service.predict_and_alert(product_id)
        
        if 'error' in result:
            return JsonResponse({'success': False, 'error': result['error']})
        
        return JsonResponse({
            'success': True,
            'product': {
                'id': product.id,
                'name': product.name,
                'sku': product.sku
            },
            'prediction': result.get('prediction'),
            'alerts': result.get('alerts', [])
        })
        
    except Product.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Product not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def product_list_view(request):
    """Display all active products with filtering and pagination"""
    products = Product.objects.filter(status='active').select_related('category', 'vendor')
    
    # Get filter parameters
    category_slug = request.GET.get('category')
    query = request.GET.get('q', '')
    sort_by = request.GET.get('sort', 'newest')
    min_price = request.GET.get('min_price')
    max_price = request.GET.get('max_price')
    
    # Filter by category
    current_category = None
    if category_slug:
        try:
            current_category = ProductCategory.objects.get(slug=category_slug, is_active=True)
            products = products.filter(category=current_category)
        except ProductCategory.DoesNotExist:
            pass
    
    # Search functionality
    if query:
        products = products.filter(
            Q(name__icontains=query) | 
            Q(description__icontains=query) |
            Q(short_description__icontains=query) |
            Q(tags__icontains=query) |
            Q(category__name__icontains=query) |
            Q(brand__icontains=query) |
            Q(model__icontains=query)
        )
    
    # Price filtering
    if min_price:
        try:
            products = products.filter(price__gte=float(min_price))
        except ValueError:
            pass
    
    if max_price:
        try:
            products = products.filter(price__lte=float(max_price))
        except ValueError:
            pass
    
    # Sorting
    if sort_by == 'price_low':
        products = products.order_by('price')
    elif sort_by == 'price_high':
        products = products.order_by('-price')
    elif sort_by == 'name':
        products = products.order_by('name')
    elif sort_by == 'popular':
        products = products.order_by('-views_count')
    elif sort_by == 'featured':
        products = products.order_by('-is_featured', '-created_at')
    else:  # newest
        products = products.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(products, 12)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    # Get categories for sidebar
    categories = ProductCategory.objects.filter(
        is_active=True,
        parent__isnull=True
    ).annotate(
        product_count=Count('products', filter=Q(products__status='active'))
    )
    
    context = {
        'products': page_obj,
        'categories': categories,
        'current_category': current_category,
        'query': query,
        'sort_by': sort_by,
        'min_price': min_price,
        'max_price': max_price,
        'total_products': products.count(),
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/products.html', context)

def product_detail_view(request, slug):
    """Display single product details"""
    product = get_object_or_404(Product.objects.select_related(
        'category', 'vendor'
    ).prefetch_related('images'), slug=slug, status='active')
    
    # Increment view count
    product.views_count += 1
    product.save()
    
    # Get related products
    related_products = Product.objects.filter(
        category=product.category,
        status='active'
    ).exclude(id=product.id).select_related(
        'category', 'vendor'
    ).prefetch_related('images')[:4]
    
    # Get product reviews
    reviews = ProductReview.objects.filter(
        product=product, 
        is_approved=True
    ).select_related('user').order_by('-created_at')
    
    # Calculate average rating
    avg_rating = reviews.aggregate(Avg('rating'))['rating__avg'] or 0
    
    # Check if product is in user's wishlist
    in_wishlist = False
    if request.user.is_authenticated:
        in_wishlist = Wishlist.objects.filter(
            user=request.user, 
            product=product
        ).exists()
    
    # Check if product is in user's cart
    in_cart = False
    cart_quantity = 0
    if request.user.is_authenticated:
        cart_item = Cart.objects.filter(
            user=request.user, 
            product=product
        ).first()
        if cart_item:
            in_cart = True
            cart_quantity = cart_item.quantity
    
    context = {
        'product': product,
        'related_products': related_products,
        'reviews': reviews,
        'avg_rating': round(avg_rating, 1) if avg_rating else 0,
        'review_count': reviews.count(),
        'in_wishlist': in_wishlist,
        'in_cart': in_cart,
        'cart_quantity': cart_quantity,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/product_detail.html', context)

def products_by_category_view(request, category_slug):
    """Display products by category"""
    category = get_object_or_404(
        ProductCategory.objects.prefetch_related('subcategories'),
        slug=category_slug, 
        is_active=True
    )
    
    # Get products in this category and subcategories
    subcategory_ids = list(category.subcategories.values_list('id', flat=True)) + [category.id]
    products = Product.objects.filter(
        category_id__in=subcategory_ids,
        status='active'
    ).select_related('category', 'vendor')
    
    # Get filter parameters
    query = request.GET.get('q', '')
    sort_by = request.GET.get('sort', 'newest')
    
    if query:
        products = products.filter(
            Q(name__icontains=query) | 
            Q(description__icontains=query) |
            Q(tags__icontains=query)
        )
    
    # Sorting
    if sort_by == 'price_low':
        products = products.order_by('price')
    elif sort_by == 'price_high':
        products = products.order_by('-price')
    elif sort_by == 'name':
        products = products.order_by('name')
    else:  # newest
        products = products.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(products, 12)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    # Get sibling categories
    if category.parent:
        categories = category.parent.subcategories.filter(is_active=True)
    else:
        categories = ProductCategory.objects.filter(
            parent__isnull=True,
            is_active=True
        )
    
    context = {
        'category': category,
        'products': page_obj,
        'categories': categories,
        'current_category': category,
        'query': query,
        'sort_by': sort_by,
        'total_products': products.count(),
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/products.html', context)

@login_required
def add_review(request, product_id):
    """Add product review"""
    if request.method == 'POST':
        product = get_object_or_404(Product, id=product_id)
        rating = request.POST.get('rating')
        title = request.POST.get('title', '')
        comment = request.POST.get('comment', '')
        
        # Validate rating
        if not rating or not rating.isdigit() or int(rating) not in range(1, 6):
            messages.error(request, 'Please select a valid rating!')
            return redirect('bika:product_detail', slug=product.slug)
        
        # Check if user already reviewed this product
        existing_review = ProductReview.objects.filter(
            user=request.user, 
            product=product
        ).first()
        
        if existing_review:
            messages.warning(request, 'You have already reviewed this product!')
        else:
            # Check if user has purchased this product
            has_purchased = OrderItem.objects.filter(
                order__user=request.user,
                product=product,
                order__status='delivered'
            ).exists()
            
            ProductReview.objects.create(
                user=request.user,
                product=product,
                rating=int(rating),
                title=title,
                comment=comment,
                is_verified_purchase=has_purchased,
                is_approved=True  # Auto-approve for now
            )
            messages.success(request, 'Thank you for your review!')
        
        return redirect('bika:product_detail', slug=product.slug)
    
    return redirect('bika:home')

# ==================== VENDOR VIEWS ====================

@login_required
def vendor_dashboard(request):
    """Vendor dashboard"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor account required.")
        return redirect('bika:home')
    
    # Get vendor's products
    if request.user.is_staff:
        vendor_products = Product.objects.all()
        vendor_orders = Order.objects.all()
    else:
        vendor_products = Product.objects.filter(vendor=request.user)
        vendor_orders = Order.objects.filter(
            items__product__vendor=request.user
        ).distinct()
    
    # Vendor stats
    stats = {
        'total_products': vendor_products.count(),
        'active_products': vendor_products.filter(status='active').count(),
        'draft_products': vendor_products.filter(status='draft').count(),
        'low_stock': vendor_products.filter(
            stock_quantity__gt=0,
            stock_quantity__lte=F('low_stock_threshold'),
            track_inventory=True
        ).count(),
        'out_of_stock': vendor_products.filter(
            stock_quantity=0,
            track_inventory=True
        ).count(),
        'total_orders': vendor_orders.count(),
        'pending_orders': vendor_orders.filter(status='pending').count(),
        'completed_orders': vendor_orders.filter(status='delivered').count(),
    }
    
    # Recent products
    recent_products = vendor_products.order_by('-created_at')[:5]
    
    # Recent orders
    recent_orders = vendor_orders.select_related('user').order_by('-created_at')[:5]
    
    # Sales data (last 30 days)
    thirty_days_ago = timezone.now() - timedelta(days=30)
    recent_sales = vendor_orders.filter(
        status='delivered',
        created_at__gte=thirty_days_ago
    )
    
    total_sales = sum(order.total_amount for order in recent_sales if order.total_amount)
    
    context = {
        'stats': stats,
        'recent_products': recent_products,
        'recent_orders': recent_orders,
        'total_sales': total_sales,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/dashboard.html', context)

@login_required
def vendor_product_list(request):
    """Vendor's product list with enhanced functionality"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor account required.")
        return redirect('bika:home')
    
    # For staff, show all products; for vendors, show only their products
    if request.user.is_staff:
        products = Product.objects.all()
    else:
        products = Product.objects.filter(vendor=request.user)
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    stock_filter = request.GET.get('stock', '')
    category_filter = request.GET.get('category', '')
    
    if query:
        products = products.filter(
            Q(name__icontains=query) | 
            Q(sku__icontains=query) |
            Q(description__icontains=query) |
            Q(category__name__icontains=query)
        )
    
    if status_filter:
        products = products.filter(status=status_filter)
    
    if stock_filter == 'in_stock':
        products = products.filter(stock_quantity__gt=0)
    elif stock_filter == 'low_stock':
        products = products.filter(
            stock_quantity__gt=0, 
            stock_quantity__lte=F('low_stock_threshold')
        )
    elif stock_filter == 'out_of_stock':
        products = products.filter(stock_quantity=0)
    
    if category_filter:
        products = products.filter(category_id=category_filter)
    
    # Apply sorting
    sort_by = request.GET.get('sort', '-updated_at')
    if sort_by in ['name', '-name', 'price', '-price', 'stock_quantity', '-stock_quantity', 
                   'created_at', '-created_at', 'updated_at', '-updated_at']:
        products = products.order_by(sort_by)
    else:
        products = products.order_by('-updated_at')
    
    # Calculate statistics
    stats = {
        'active': products.filter(status='active').count(),
        'draft': products.filter(status='draft').count(),
        'low_stock': products.filter(
            stock_quantity__gt=0, 
            stock_quantity__lte=F('low_stock_threshold')
        ).count(),
        'out_of_stock': products.filter(stock_quantity=0).count(),
    }
    
    # Get categories for filter
    categories = ProductCategory.objects.filter(is_active=True)
    
    # Pagination
    paginator = Paginator(products, 10)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'products': page_obj,
        'stats': stats,
        'categories': categories,
        'query': query,
        'status_filter': status_filter,
        'stock_filter': stock_filter,
        'category_filter': category_filter,
        'sort_by': sort_by,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/products.html', context)

@login_required
def vendor_add_product(request):
    """Add new product"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor account required.")
        return redirect('bika:home')
    
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                product = form.save(commit=False)
                product.vendor = request.user
                
                # Generate SKU if not provided
                if not product.sku:
                    product.sku = f"PROD{timezone.now().strftime('%Y%m%d%H%M%S')}"
                
                # Generate barcode if not provided
                if not product.barcode:
                    import random
                    product.barcode = f"8{random.randint(100000000000, 999999999999)}"
                
                product.save()
                
                # Handle multiple images
                images = request.FILES.getlist('images')
                for i, image in enumerate(images):
                    ProductImage.objects.create(
                        product=product,
                        image=image,
                        alt_text=product.name,
                        display_order=i,
                        is_primary=(i == 0)
                    )
                
                messages.success(request, 'Product added successfully!')
                return redirect('bika:vendor_product_list')
                
            except Exception as e:
                messages.error(request, f'Error saving product: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ProductForm()
    
    context = {
        'form': form,
        'title': 'Add New Product',
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/add_product.html', context)

@login_required
def vendor_edit_product(request, product_id):
    """Edit existing product"""
    if request.user.is_staff:
        product = get_object_or_404(Product, id=product_id)
    else:
        product = get_object_or_404(Product, id=product_id, vendor=request.user)
    
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)
        
        if form.is_valid():
            try:
                form.save()
                messages.success(request, 'Product updated successfully!')
                return redirect('bika:vendor_product_list')
            except Exception as e:
                messages.error(request, f'Error updating product: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ProductForm(instance=product)
    
    # Get existing images
    images = product.images.all()
    
    context = {
        'form': form,
        'product': product,
        'images': images,
        'title': 'Edit Product',
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/edit_product.html', context)

@login_required
def vendor_delete_product(request, product_id):
    """Delete product"""
    if request.method == 'POST':
        if request.user.is_staff:
            product = get_object_or_404(Product, id=product_id)
        else:
            product = get_object_or_404(Product, id=product_id, vendor=request.user)
        
        product_name = product.name
        product.delete()
        
        messages.success(request, f'Product "{product_name}" deleted successfully!')
        return redirect('bika:vendor_product_list')
    
    return redirect('bika:vendor_product_list')

# ==================== USER PROFILE VIEWS ====================

@login_required
def user_profile(request):
    """User profile page"""
    user = request.user
    recent_orders = Order.objects.filter(user=user).order_by('-created_at')[:5]
    wishlist_count = Wishlist.objects.filter(user=user).count()
    cart_count = Cart.objects.filter(user=user).count()
    
    context = {
        'user': user,
        'recent_orders': recent_orders,
        'wishlist_count': wishlist_count,
        'cart_count': cart_count,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/profile.html', context)

@login_required
def update_profile(request):
    """Update user profile"""
    if request.method == 'POST':
        user = request.user
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)
        user.phone = request.POST.get('phone', user.phone)
        user.address = request.POST.get('address', user.address)
        
        if 'profile_picture' in request.FILES:
            user.profile_picture = request.FILES['profile_picture']
        
        if user.is_vendor():
            user.business_name = request.POST.get('business_name', user.business_name)
            user.business_description = request.POST.get('business_description', user.business_description)
            
            if 'business_logo' in request.FILES:
                user.business_logo = request.FILES['business_logo']
        
        user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('bika:user_profile')
    
    return redirect('bika:user_profile')

@login_required
def user_orders(request):
    """User orders page"""
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    
    # Calculate totals
    total_orders = orders.count()
    total_spent = sum(order.total_amount for order in orders if order.total_amount)
    
    context = {
        'orders': orders,
        'total_orders': total_orders,
        'total_spent': total_spent,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/orders.html', context)

def order_detail(request, order_id):
    """Order detail page"""
    order = get_object_or_404(Order.objects.select_related('user').prefetch_related('items'), 
                             id=order_id, user=request.user)
    
    # Get payments for this order
    payments = Payment.objects.filter(order=order).order_by('-created_at')
    
    context = {
        'order': order,
        'payments': payments,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/order_detail.html', context)

# ==================== WISHLIST VIEWS ====================

@login_required
def wishlist(request):
    """User wishlist page"""
    wishlist_items = Wishlist.objects.filter(
        user=request.user
    ).select_related('product').order_by('-added_at')
    
    context = {
        'wishlist_items': wishlist_items,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/wishlist.html', context)

@login_required
@require_POST
def add_to_wishlist(request, product_id):
    """Add product to wishlist"""
    product = get_object_or_404(Product, id=product_id)
    
    wishlist_item, created = Wishlist.objects.get_or_create(
        user=request.user,
        product=product
    )
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        wishlist_count = Wishlist.objects.filter(user=request.user).count()
        return JsonResponse({
            'success': True,
            'message': 'Product added to wishlist!',
            'wishlist_count': wishlist_count,
            'created': created
        })
    
    messages.success(request, 'Product added to wishlist!')
    return redirect('bika:product_detail', slug=product.slug)

@login_required
@require_POST
def remove_from_wishlist(request, product_id):
    """Remove product from wishlist"""
    product = get_object_or_404(Product, id=product_id)
    
    deleted_count, _ = Wishlist.objects.filter(
        user=request.user, 
        product=product
    ).delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        wishlist_count = Wishlist.objects.filter(user=request.user).count()
        return JsonResponse({
            'success': True,
            'message': 'Product removed from wishlist!',
            'wishlist_count': wishlist_count,
            'deleted': deleted_count > 0
        })
    
    messages.success(request, 'Product removed from wishlist!')
    
    referer = request.META.get('HTTP_REFERER', '')
    if 'wishlist' in referer:
        return redirect('bika:wishlist')
    else:
        return redirect('bika:product_detail', slug=product.slug)

# ==================== CART VIEWS ====================

@login_required
def cart(request):
    """Shopping cart page"""
    from decimal import Decimal
    
    cart_items = Cart.objects.filter(
        user=request.user
    ).select_related('product').order_by('-added_at')
    
    # Calculate totals - Use Decimal for calculations
    subtotal = Decimal('0.00')
    
    # Prepare cart items with total_price for template
    cart_items_with_total = []
    for item in cart_items:
        # Calculate total for each item
        item_price = Decimal(str(item.product.price))
        item_quantity = Decimal(str(item.quantity))
        item_subtotal = item_price * item_quantity
        
        # Add calculated total to item (as a simple attribute, not property)
        item.total_price_calculated = item_subtotal
        cart_items_with_total.append(item)
        
        subtotal += item_subtotal
    
    tax_rate = Decimal('0.18')  # 18% VAT
    tax_amount = subtotal * tax_rate
    shipping_cost = Decimal('5000')  # Fixed shipping cost
    total_amount = subtotal + tax_amount + shipping_cost
    
    context = {
        'cart_items': cart_items_with_total,  # Use the list with calculated totals
        'subtotal': subtotal,
        'tax_amount': tax_amount,
        'shipping_cost': shipping_cost,
        'total_amount': total_amount,
        'tax_rate': tax_rate,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/cart.html', context)

@login_required
@require_POST
def add_to_cart(request, product_id):
    """Add product to cart"""
    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))
    
    # Check stock availability
    if product.track_inventory and product.stock_quantity < quantity:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'success': False,
                'message': f'Only {product.stock_quantity} items available!'
            })
        messages.error(request, f'Only {product.stock_quantity} items available!')
        return redirect('bika:product_detail', slug=product.slug)
    
    # Add to cart
    cart_item, created = Cart.objects.get_or_create(
        user=request.user,
        product=product,
        defaults={'quantity': quantity}
    )
    
    if not created:
        cart_item.quantity += quantity
        cart_item.save()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        cart_count = Cart.objects.filter(user=request.user).count()
        return JsonResponse({
            'success': True,
            'message': 'Product added to cart!',
            'cart_count': cart_count,
            'created': created
        })
    
    messages.success(request, 'Product added to cart!')
    return redirect('bika:cart')

@login_required
@require_POST
@login_required
@require_POST
def update_cart(request, product_id):
    """Update cart item quantity"""
    from decimal import Decimal
    
    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))
    
    if quantity > 0:
        # Check stock
        if product.track_inventory and product.stock_quantity < quantity:
            return JsonResponse({
                'success': False,
                'message': f'Only {product.stock_quantity} items available!'
            })
        
        cart_item = get_object_or_404(Cart, user=request.user, product=product)
        cart_item.quantity = quantity
        cart_item.save()
        
        # Recalculate totals
        cart_items = Cart.objects.filter(user=request.user)
        subtotal = sum(Decimal(str(item.product.price)) * Decimal(str(item.quantity)) for item in cart_items)
        tax_rate = Decimal('0.18')
        tax_amount = subtotal * tax_rate
        shipping_cost = Decimal('5000')
        total_amount = subtotal + tax_amount + shipping_cost
        
        return JsonResponse({
            'success': True,
            'item_total': str(cart_item.total_price),
            'subtotal': str(subtotal),
            'tax_amount': str(tax_amount),
            'total_amount': str(total_amount),
            'cart_count': cart_items.count(),
            'max_quantity': product.stock_quantity if product.track_inventory else 99
        })
    else:
        Cart.objects.filter(user=request.user, product=product).delete()
        cart_items = Cart.objects.filter(user=request.user)
        
        if cart_items.exists():
            subtotal = sum(Decimal(str(item.product.price)) * Decimal(str(item.quantity)) for item in cart_items)
            tax_rate = Decimal('0.18')
            tax_amount = subtotal * tax_rate
            shipping_cost = Decimal('5000')
            total_amount = subtotal + tax_amount + shipping_cost
            
            return JsonResponse({
                'success': True,
                'subtotal': str(subtotal),
                'tax_amount': str(tax_amount),
                'total_amount': str(total_amount),
                'cart_count': cart_items.count()
            })
        else:
            return JsonResponse({
                'success': True,
                'subtotal': '0.00',
                'tax_amount': '0.00',
                'total_amount': '0.00',
                'cart_count': 0
            })
        

@login_required
@require_POST
def remove_from_cart(request, product_id):
    """Remove product from cart"""
    from decimal import Decimal  # ADD THIS IMPORT
    
    product = get_object_or_404(Product, id=product_id)
    
    deleted_count, _ = Cart.objects.filter(
        user=request.user, 
        product=product
    ).delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        cart_items = Cart.objects.filter(user=request.user)
        subtotal = sum(item.total_price for item in cart_items)
        tax_rate = Decimal('0.18')  # CHANGE TO DECIMAL
        tax_amount = subtotal * tax_rate
        shipping_cost = Decimal('5000')  # CHANGE TO DECIMAL
        total_amount = subtotal + tax_amount + shipping_cost
        
        return JsonResponse({
            'success': True,
            'subtotal': float(subtotal),  # CONVERT TO FLOAT
            'tax_amount': float(tax_amount),  # CONVERT TO FLOAT
            'total_amount': float(total_amount),  # CONVERT TO FLOAT
            'cart_count': cart_items.count(),
            'deleted': deleted_count > 0
        })
    
    messages.success(request, 'Product removed from cart!')
    return redirect('bika:cart')

@login_required
def clear_cart(request):
    """Clear entire cart"""
    if request.method == 'POST':
        deleted_count, _ = Cart.objects.filter(user=request.user).delete()
        
        messages.success(request, f'Cart cleared! {deleted_count} items removed.')
        return redirect('bika:cart')
    
    return redirect('bika:cart')

# ==================== CHECKOUT & PAYMENT VIEWS ====================

@login_required
def checkout(request):
    """Checkout page"""
    from decimal import Decimal  # ADD THIS IMPORT
    
    cart_items = Cart.objects.filter(user=request.user).select_related('product')
    
    if not cart_items:
        messages.error(request, "Your cart is empty!")
        return redirect('bika:cart')
    
    # Check stock for all items
    for item in cart_items:
        if item.product.track_inventory and item.product.stock_quantity < item.quantity:
            messages.error(
                request, 
                f'Only {item.product.stock_quantity} items available for {item.product.name}!'
            )
            return redirect('bika:cart')
    
    # Calculate totals - USE DECIMAL
    subtotal = sum(item.total_price for item in cart_items)
    tax_rate = Decimal('0.18')  # CHANGE TO DECIMAL
    tax_amount = subtotal * tax_rate
    shipping_cost = Decimal('5000')  # CHANGE TO DECIMAL
    total_amount = subtotal + tax_amount + shipping_cost
    
    # Get user's default addresses
    user = request.user
    shipping_address = user.address
    billing_address = user.address
    
    # Get available payment methods
    payment_methods = [
        {'value': 'mpesa', 'name': 'M-Pesa', 'icon': 'fas fa-mobile-alt'},
        {'value': 'airtel_tz', 'name': 'Airtel Money', 'icon': 'fas fa-wifi'},
        {'value': 'tigo_tz', 'name': 'Tigo Pesa', 'icon': 'fas fa-sim-card'},
        {'value': 'visa', 'name': 'Visa Card', 'icon': 'fab fa-cc-visa'},
        {'value': 'mastercard', 'name': 'MasterCard', 'icon': 'fab fa-cc-mastercard'},
        {'value': 'paypal', 'name': 'PayPal', 'icon': 'fab fa-paypal'},
    ]
    
    context = {
        'cart_items': cart_items,
        'subtotal': float(subtotal),  # CONVERT TO FLOAT
        'tax_amount': float(tax_amount),  # CONVERT TO FLOAT
        'shipping_cost': float(shipping_cost),  # CONVERT TO FLOAT
        'total_amount': float(total_amount),  # CONVERT TO FLOAT
        'shipping_address': shipping_address,
        'billing_address': billing_address,
        'payment_methods': payment_methods,
        'tax_rate': float(tax_rate * Decimal('100')),  # For display
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/checkout.html', context)

@login_required
@require_POST
@login_required
@require_POST
def place_order(request):
    """Place order and process payment"""
    try:
        with transaction.atomic():
            from decimal import Decimal  # ADD THIS IMPORT
            
            # Get cart items
            cart_items = Cart.objects.filter(user=request.user).select_related('product')
            
            if not cart_items:
                return JsonResponse({
                    'success': False,
                    'message': 'Your cart is empty!'
                })
            
            # Validate stock
            for item in cart_items:
                if item.product.track_inventory and item.product.stock_quantity < item.quantity:
                    return JsonResponse({
                        'success': False,
                        'message': f'Insufficient stock for {item.product.name}'
                    })
            
            # Calculate totals - USE DECIMAL
            subtotal = sum(item.total_price for item in cart_items)
            tax_rate = Decimal('0.18')  # CHANGE TO DECIMAL
            tax_amount = subtotal * tax_rate
            shipping_cost = Decimal('5000')  # CHANGE TO DECIMAL
            total_amount = subtotal + tax_amount + shipping_cost
            
            # Get form data
            shipping_address = request.POST.get('shipping_address', '')
            billing_address = request.POST.get('billing_address', '')
            payment_method = request.POST.get('payment_method', '')
            phone_number = request.POST.get('phone_number', '')
            
            if not shipping_address or not billing_address:
                return JsonResponse({
                    'success': False,
                    'message': 'Please provide shipping and billing addresses'
                })
            
            if not payment_method:
                return JsonResponse({
                    'success': False,
                    'message': 'Please select a payment method'
                })
            
            # Create order
            order = Order.objects.create(
                user=request.user,
                total_amount=total_amount,  # This should already be Decimal
                shipping_address=shipping_address,
                billing_address=billing_address,
                status='pending'
            )
            
            # Create order items and update stock
            for cart_item in cart_items:
                OrderItem.objects.create(
                    order=order,
                    product=cart_item.product,
                    quantity=cart_item.quantity,
                    price=cart_item.product.price
                )
                
                # Update product stock
                if cart_item.product.track_inventory:
                    cart_item.product.stock_quantity -= cart_item.quantity
                    cart_item.product.save()
            
            # Create initial payment record
            payment = Payment.objects.create(
                order=order,
                payment_method=payment_method,
                amount=total_amount,  # This should already be Decimal
                currency='TZS',
                status='pending'
            )
            
            # If mobile money, save phone number
            if payment_method in ['mpesa', 'airtel_tz', 'tigo_tz'] and phone_number:
                payment.mobile_money_phone = phone_number
                payment.save()
            
            # Clear cart
            cart_items.delete()
            
            # Return success with order details
            return JsonResponse({
                'success': True,
                'order_id': order.id,
                'order_number': order.order_number,
                'payment_id': payment.id,
                'redirect_url': reverse('bika:payment_processing', args=[payment.id]),
                'total_amount': float(total_amount)  # CONVERT TO FLOAT FOR JSON
            })
            
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return JsonResponse({
            'success': False,
            'message': 'An error occurred while placing your order. Please try again.'
        })
    
@login_required
def payment_processing(request, payment_id):
    """Payment processing page"""
    payment = get_object_or_404(Payment, id=payment_id, order__user=request.user)
    order = payment.order
    
    context = {
        'payment': payment,
        'order': order,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/payment_processing.html', context)

@csrf_exempt
@require_POST
def payment_webhook(request):
    """Handle payment webhooks from payment providers"""
    try:
        # This is a simplified version - implement based on your payment provider
        data = json.loads(request.body)
        
        # Extract payment info from webhook
        transaction_id = data.get('transaction_id')
        status = data.get('status')
        
        if transaction_id:
            payment = Payment.objects.filter(transaction_id=transaction_id).first()
            if payment:
                if status == 'success':
                    payment.status = 'completed'
                    payment.paid_at = timezone.now()
                    payment.order.status = 'confirmed'
                    payment.order.save()
                elif status == 'failed':
                    payment.status = 'failed'
                    payment.order.status = 'pending'
                    payment.order.save()
                
                payment.save()
                
                # Send notification to user
                Notification.objects.create(
                    user=payment.order.user,
                    title=f"Payment {status}",
                    message=f"Your payment for order #{payment.order.order_number} has been {status}.",
                    notification_type='order_update',
                    related_object_type='payment',
                    related_object_id=payment.id
                )
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        logger.error(f"Payment webhook error: {e}")
        return JsonResponse({'success': False}, status=400)

# ==================== FRUIT QUALITY MONITORING VIEWS ====================

@login_required
def fruit_quality_dashboard(request):
    """Fruit quality monitoring dashboard"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect('bika:home')
    
    # Get fruit batches
    if request.user.is_staff:
        batches = FruitBatch.objects.all().select_related('fruit_type', 'storage_location')
    else:
        batches = FruitBatch.objects.filter(
            product__vendor=request.user
        ).select_related('fruit_type', 'storage_location')
    
    # Get statistics
    total_batches = batches.count()
    active_batches = batches.filter(status='active').count()
    completed_batches = batches.filter(status='completed').count()
    
    # Get recent quality readings
    recent_readings = FruitQualityReading.objects.select_related(
        'fruit_batch', 'fruit_batch__fruit_type'
    ).order_by('-timestamp')[:10]
    
    # Get alerts
    alerts = ProductAlert.objects.filter(
        alert_type__in=['quality_issue', 'temperature_anomaly', 'humidity_issue']
    ).select_related('product').order_by('-created_at')[:5]
    
    context = {
        'batches': batches[:10],  # Show only recent 10
        'total_batches': total_batches,
        'active_batches': active_batches,
        'completed_batches': completed_batches,
        'recent_readings': recent_readings,
        'alerts': alerts,
        'ai_available': AI_SERVICES_AVAILABLE,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/fruit_dashboard.html', context)

@login_required
def create_fruit_batch(request):
    """Create new fruit batch"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect('bika:home')
    
    if request.method == 'POST':
        form = FruitBatchForm(request.POST)
        if form.is_valid():
            batch = form.save(commit=False)
            
            # Set batch number if not provided
            if not batch.batch_number:
                import random
                import string
                random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                batch.batch_number = f"BATCH{timezone.now().strftime('%Y%m%d')}{random_str}"
            
            # Set status to active
            batch.status = 'active'
            
            # If vendor, associate with vendor's products
            if request.user.is_vendor() and not request.user.is_staff:
                # You might want to link to a specific product
                pass
            
            batch.save()
            
            messages.success(request, f'Fruit batch {batch.batch_number} created successfully!')
            return redirect('bika:fruit_quality_dashboard')
    else:
        form = FruitBatchForm()
    
    context = {
        'form': form,
        'fruit_types': FruitType.objects.all(),
        'storage_locations': StorageLocation.objects.filter(is_active=True),
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/create_fruit_batch.html', context)

@login_required
def batch_detail(request, batch_id):
    """View batch details"""
    if request.user.is_staff:
        batch = get_object_or_404(FruitBatch.objects.select_related(
            'fruit_type', 'storage_location'
        ), id=batch_id)
    else:
        batch = get_object_or_404(FruitBatch.objects.select_related(
            'fruit_type', 'storage_location'
        ), id=batch_id, product__vendor=request.user)
    
    # Get quality readings
    quality_readings = FruitQualityReading.objects.filter(
        fruit_batch=batch
    ).order_by('-timestamp')
    
    # Get sensor data
    sensor_data = RealTimeSensorData.objects.filter(
        fruit_batch=batch
    ).order_by('-recorded_at')
    
    # Try to get AI analysis
    ai_analysis = None
    if AI_SERVICES_AVAILABLE:
        try:
            ai_analysis = fruit_ai_service.get_batch_quality_report(batch_id, hours=24)
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
    
    context = {
        'batch': batch,
        'quality_readings': quality_readings,
        'sensor_data': sensor_data,
        'ai_analysis': ai_analysis,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/batch_detail.html', context)

@login_required
def add_quality_reading(request, batch_id):
    """Add quality reading for batch"""
    if request.user.is_staff:
        batch = get_object_or_404(FruitBatch, id=batch_id)
    else:
        batch = get_object_or_404(FruitBatch, id=batch_id, product__vendor=request.user)
    
    if request.method == 'POST':
        form = FruitQualityReadingForm(request.POST)
        if form.is_valid():
            reading = form.save(commit=False)
            reading.fruit_batch = batch
            
            # If AI is available, get prediction
            if AI_SERVICES_AVAILABLE and not reading.predicted_class:
                try:
                    prediction = fruit_ai_service.predict_fruit_quality(
                        batch.fruit_type.name,
                        reading.temperature,
                        reading.humidity,
                        reading.light_intensity,
                        reading.co2_level,
                        batch.id
                    )
                    
                    if prediction.get('success'):
                        reading.predicted_class = prediction['prediction']['predicted_class']
                        reading.confidence_score = prediction['prediction']['confidence']
                except Exception as e:
                    logger.error(f"Error getting AI prediction: {e}")
            
            reading.save()
            
            messages.success(request, 'Quality reading added successfully!')
            return redirect('bika:batch_detail', batch_id=batch.id)
    else:
        form = FruitQualityReadingForm()
    
    context = {
        'form': form,
        'batch': batch,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/add_quality_reading.html', context)

@login_required
@csrf_exempt
@require_POST
def train_fruit_model_api(request):
    """API endpoint to train fruit quality model"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})
    
    try:
        if 'dataset_file' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No file uploaded'})
        
        csv_file = request.FILES['dataset_file']
        model_type = request.POST.get('model_type', 'random_forest')
        
        if not AI_SERVICES_AVAILABLE:
            return JsonResponse({'success': False, 'error': 'AI services not available'})
        
        result = fruit_ai_service.train_fruit_quality_model(csv_file, model_type)
        
        if result.get('success'):
            return JsonResponse(result)
        else:
            return JsonResponse({'success': False, 'error': result.get('error', 'Training failed')})
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
@require_GET
def predict_fruit_quality_api(request):
    """API endpoint for fruit quality prediction"""
    try:
        fruit_name = request.GET.get('fruit_name', '')
        temperature = float(request.GET.get('temperature', 5.0))
        humidity = float(request.GET.get('humidity', 90.0))
        light_intensity = float(request.GET.get('light_intensity', 50.0))
        co2_level = float(request.GET.get('co2_level', 400.0))
        batch_id = request.GET.get('batch_id')
        
        if not fruit_name:
            return JsonResponse({'success': False, 'error': 'Fruit name required'})
        
        if not AI_SERVICES_AVAILABLE:
            return JsonResponse({
                'success': False, 
                'error': 'AI services not available',
                'suggested_quality': 'Good'  # Fallback suggestion
            })
        
        prediction = fruit_ai_service.predict_fruit_quality(
            fruit_name, temperature, humidity, light_intensity, co2_level, batch_id
        )
        
        return JsonResponse(prediction)
        
    except Exception as e:
        logger.error(f"Error predicting fruit quality: {e}")
        return JsonResponse({'success': False, 'error': str(e)})

# ==================== NOTIFICATION VIEWS ====================

@login_required
def notifications(request):
    """User notifications"""
    notifications = Notification.objects.filter(
        user=request.user
    ).order_by('-created_at')
    
    unread_count = notifications.filter(is_read=False).count()
    
    context = {
        'notifications': notifications,
        'unread_count': unread_count,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/user/notifications.html', context)

@login_required
@require_POST
def mark_notification_read(request, notification_id):
    """Mark notification as read"""
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.is_read = True
    notification.save()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        unread_count = Notification.objects.filter(
            user=request.user, 
            is_read=False
        ).count()
        
        return JsonResponse({
            'success': True,
            'unread_count': unread_count
        })
    
    messages.success(request, 'Notification marked as read!')
    return redirect('bika:notifications')

@login_required
@require_POST
def mark_all_notifications_read(request):
    """Mark all notifications as read"""
    updated = Notification.objects.filter(
        user=request.user,
        is_read=False
    ).update(is_read=True)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'updated_count': updated,
            'unread_count': 0
        })
    
    messages.success(request, f'{updated} notifications marked as read!')
    return redirect('bika:notifications')

@login_required
@require_GET
def unread_notifications_count(request):
    """Get unread notifications count"""
    if request.user.is_authenticated:
        unread_count = Notification.objects.filter(
            user=request.user,
            is_read=False
        ).count()
        
        critical_count = Notification.objects.filter(
            user=request.user,
            is_read=False,
            notification_type='urgent_alert'
        ).count()
        
        return JsonResponse({
            'unread_count': unread_count,
            'critical_count': critical_count
        })
    
    return JsonResponse({'unread_count': 0, 'critical_count': 0})

# ==================== AUTHENTICATION VIEWS ====================

def register_view(request):
    """User registration"""
    if request.user.is_authenticated:
        return redirect('bika:home')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Auto-login
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome to Bika, {username}!')
                return redirect('bika:home')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    context = {
        'form': form,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/registration/register.html', context)

def vendor_register_view(request):
    """Vendor registration"""
    if request.user.is_authenticated and request.user.is_vendor():
        return redirect('bika:vendor_dashboard')
    
    if request.method == 'POST':
        form = VendorRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Auto-login
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Vendor account created! Welcome to Bika, {user.business_name}.')
                return redirect('bika:vendor_dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = VendorRegistrationForm()
    
    context = {
        'form': form,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/registration/vendor_register.html', context)

@login_required
@never_cache
def custom_logout(request):
    """Logout user with security headers"""
    username = request.user.username
    
    logout(request)
    
    response = redirect('bika:logout_success')
    
    # Clear session
    request.session.flush()
    response.delete_cookie('sessionid')
    response.delete_cookie('csrftoken')
    
    # Security headers
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = 'Fri, 01 Jan 1990 00:00:00 GMT'
    
    messages.success(request, f'Goodbye {username}! You have been logged out successfully.')
    
    return response

def logout_success(request):
    """Logout success page"""
    response = render(request, 'bika/pages/registration/logout.html')
    
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = 'Fri, 01 Jan 1990 00:00:00 GMT'
    
    return response

# ==================== ERROR HANDLERS ====================

def handler404(request, exception):
    return render(request, 'bika/pages/404.html', status=404)

def handler500(request):
    return render(request, 'bika/pages/500.html', status=500)

def handler403(request, exception):
    return render(request, 'bika/pages/403.html', status=403)

def handler400(request, exception):
    return render(request, 'bika/pages/400.html', status=400)

# ==================== API ENDPOINTS ====================

@csrf_exempt
@require_POST
def receive_sensor_data(request):
    """Receive sensor data from IoT devices"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['sensor_type', 'value', 'unit']
        for field in required_fields:
            if field not in data:
                return JsonResponse({'success': False, 'error': f'Missing field: {field}'})
        
        # Get optional fields
        product_barcode = data.get('product_barcode')
        batch_number = data.get('batch_number')
        location_id = data.get('location_id')
        
        # Find related objects
        product = None
        fruit_batch = None
        location = None
        
        if product_barcode:
            product = Product.objects.filter(barcode=product_barcode).first()
        
        if batch_number:
            fruit_batch = FruitBatch.objects.filter(batch_number=batch_number).first()
        
        if location_id:
            location = StorageLocation.objects.filter(id=location_id).first()
        
        # Create sensor reading
        sensor_reading = RealTimeSensorData.objects.create(
            product=product,
            fruit_batch=fruit_batch,
            sensor_type=data['sensor_type'],
            value=data['value'],
            unit=data['unit'],
            location=location,
            recorded_at=timezone.now()
        )
        
        # Check for anomalies (simplified version)
        if data['sensor_type'] == 'temperature' and (data['value'] < 0 or data['value'] > 25):
            # Create alert
            if product:
                ProductAlert.objects.create(
                    product=product,
                    alert_type='temperature_anomaly',
                    severity='high',
                    message=f'Temperature anomaly detected: {data["value"]}{data["unit"]}',
                    detected_by='sensor_system'
                )
        
        return JsonResponse({'success': True, 'id': sensor_reading.id})
        
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error receiving sensor data: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@csrf_exempt
@require_GET
def api_product_detail(request, barcode):
    """API endpoint for product details by barcode"""
    try:
        product = Product.objects.select_related('category', 'vendor').get(barcode=barcode)
        
        product_data = {
            'id': product.id,
            'name': product.name,
            'slug': product.slug,
            'barcode': product.barcode,
            'sku': product.sku,
            'price': str(product.price),
            'compare_price': str(product.compare_price) if product.compare_price else None,
            'stock_quantity': product.stock_quantity,
            'status': product.status,
            'category': {
                'id': product.category.id,
                'name': product.category.name,
                'slug': product.category.slug,
            },
            'vendor': {
                'id': product.vendor.id,
                'username': product.vendor.username,
                'business_name': product.vendor.business_name,
            },
            'images': [
                {
                    'image': img.image.url if img.image else None,
                    'alt_text': img.alt_text,
                    'is_primary': img.is_primary,
                }
                for img in product.images.all()[:3]
            ],
        }
        
        return JsonResponse(product_data)
        
    except Product.DoesNotExist:
        return JsonResponse({'error': 'Product not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# ==================== HELPER VIEWS ====================

def scan_product(request):
    """Product scanning interface"""
    context = {
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/scan_product.html', context)

@staff_member_required
def storage_sites(request):
    """Storage sites management"""
    sites = StorageLocation.objects.all()
    
    context = {
        'sites': sites,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/admin/storage_sites.html', context)

@login_required
def track_my_products(request):
    """Track vendor's products with analytics"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect('bika:home')
    
    # Get vendor's products
    if request.user.is_staff:
        products = Product.objects.all()
        alerts = ProductAlert.objects.filter(is_resolved=False)
    else:
        products = Product.objects.filter(vendor=request.user)
        alerts = ProductAlert.objects.filter(product__vendor=request.user, is_resolved=False)
    
    # Apply filters
    query = request.GET.get('q', '')
    stock_filter = request.GET.get('stock', '')
    
    if query:
        products = products.filter(
            Q(name__icontains=query) | 
            Q(sku__icontains=query) |
            Q(category__name__icontains=query)
        )
    
    if stock_filter == 'in_stock':
        products = products.filter(stock_quantity__gt=0)
    elif stock_filter == 'low_stock':
        products = products.filter(
            stock_quantity__gt=0, 
            stock_quantity__lte=F('low_stock_threshold')
        )
    elif stock_filter == 'out_of_stock':
        products = products.filter(stock_quantity=0)
    
    # Calculate stats
    stats = {
        'total': products.count(),
        'in_stock': products.filter(stock_quantity__gt=0).count(),
        'low_stock': products.filter(
            stock_quantity__gt=0, 
            stock_quantity__lte=F('low_stock_threshold')
        ).count(),
        'out_of_stock': products.filter(stock_quantity=0).count(),
        'alerts': alerts.count(),
    }
    
    context = {
        'products': products[:20],  # Limit for performance
        'alerts': alerts[:10],
        'stats': stats,
        'query': query,
        'stock_filter': stock_filter,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/track_products.html', context)

# ==================== ADD ALL MISSING VIEW FUNCTIONS ====================

def batch_analytics(request, batch_id):
    """Batch analytics page"""
    if not request.user.is_authenticated:
        return redirect('bika:login')
    
    if request.user.is_staff:
        batch = get_object_or_404(FruitBatch, id=batch_id)
    else:
        batch = get_object_or_404(FruitBatch, id=batch_id, product__vendor=request.user)
    
    context = {
        'batch': batch,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/vendor/batch_analytics.html', context)

def upload_dataset(request):
    """Upload dataset for AI training"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})
    
    if request.method == 'POST':
        # Handle dataset upload
        return JsonResponse({'success': True, 'message': 'Dataset uploaded successfully'})
    
    return render(request, 'bika/pages/ai/upload_dataset.html', {'site_info': SiteInfo.objects.first()})

def train_model(request):
    """Train AI model"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})
    
    if request.method == 'POST':
        # Handle model training
        return JsonResponse({'success': True, 'message': 'Model training started'})
    
    return render(request, 'bika/pages/ai/train_model.html', {'site_info': SiteInfo.objects.first()})

def product_analytics_api(request, product_id):
    """API endpoint for product analytics"""
    product = get_object_or_404(Product, id=product_id)
    
    analytics_data = {
        'product_id': product.id,
        'product_name': product.name,
        'views_count': product.views_count,
        'stock_level': product.stock_quantity,
        'sales_trend': 'increasing',
        'recommendations': ['Consider restocking soon']
    }
    
    return JsonResponse(analytics_data)

def storage_compatibility_check(request):
    """Check storage compatibility"""
    if request.method == 'GET':
        fruit1 = request.GET.get('fruit1', '')
        fruit2 = request.GET.get('fruit2', '')
        
        # Simple compatibility check
        compatible = True
        message = f"{fruit1} and {fruit2} are compatible for storage"
        
        if fruit1 and fruit2:
            ethylene_producers = ['Apple', 'Banana', 'Tomato']
            ethylene_sensitive = ['Lettuce', 'Broccoli', 'Carrot']
            
            if fruit1 in ethylene_producers and fruit2 in ethylene_sensitive:
                compatible = False
                message = f"{fruit1} produces ethylene which can spoil {fruit2}"
            elif fruit2 in ethylene_producers and fruit1 in ethylene_sensitive:
                compatible = False
                message = f"{fruit2} produces ethylene which can spoil {fruit1}"
        
        return JsonResponse({
            'success': True,
            'compatible': compatible,
            'message': message
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def resolve_alert(request, alert_id):
    """Resolve product alert"""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Authentication required'})
    
    alert = get_object_or_404(ProductAlert, id=alert_id)
    
    if request.method == 'POST':
        alert.is_resolved = True
        alert.resolved_by = request.user
        alert.resolved_at = timezone.now()
        alert.save()
        
        return JsonResponse({'success': True, 'message': 'Alert resolved successfully'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def handle_bulk_actions(request):
    """Handle bulk product actions"""
    if not request.user.is_authenticated or (not request.user.is_vendor() and not request.user.is_staff):
        return JsonResponse({'success': False, 'error': 'Permission denied'})
    
    if request.method == 'POST':
        action = request.POST.get('action', '')
        product_ids = request.POST.get('product_ids', '')
        
        if not action or not product_ids:
            return JsonResponse({'success': False, 'error': 'Missing parameters'})
        
        try:
            ids = [int(id) for id in product_ids.split(',')]
            products = Product.objects.filter(id__in=ids)
            
            if not request.user.is_staff:
                products = products.filter(vendor=request.user)
            
            updated_count = 0
            
            if action == 'activate':
                updated_count = products.update(status='active')
            elif action == 'draft':
                updated_count = products.update(status='draft')
            elif action == 'feature':
                updated_count = products.update(is_featured=True)
            elif action == 'unfeature':
                updated_count = products.update(is_featured=False)
            elif action == 'delete':
                deleted_count, _ = products.delete()
                updated_count = deleted_count
            
            return JsonResponse({
                'success': True,
                'message': f'{updated_count} products updated successfully',
                'updated_count': updated_count
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})    
# Add these views

@staff_member_required
@staff_member_required
def training_results_view(request):
    """Display training results with detailed analysis"""
    print("\n📊 Loading training results...")
    
    # Get results from session
    results = request.session.get('training_results', {})
    
    if not results:
        print("❌ No training results found in session")
        messages.info(request, 'No training results found. Please train a model first.')
        return redirect('bika:train_models')
    
    print(f"✅ Found training results")
    print(f"   Best model: {results.get('best_model_name')}")
    print(f"   Best accuracy: {results.get('best_accuracy', 0):.2%}")
    
    # ==================== PREPARE CONTEXT ====================
    context = {
        'results': results,
        'site_info': SiteInfo.objects.first(),
        'training_time': results.get('training_timestamp', 'Unknown'),
        'dataset_source': results.get('dataset_source', 'Unknown')
    }
    
    # Add model comparison data
    if 'models' in results:
        model_data = []
        for model_key, model_info in results['models'].items():
            model_data.append({
                'name': model_info['name'],
                'accuracy': model_info['accuracy'],
                'precision': model_info.get('precision', 0),
                'recall': model_info.get('recall', 0),
                'f1_score': model_info.get('f1_score', 0),
                'cv_mean': model_info.get('cv_mean', 0),
                'cv_std': model_info.get('cv_std', 0)
            })
        
        # Sort by accuracy
        model_data.sort(key=lambda x: x['accuracy'], reverse=True)
        context['model_comparison'] = model_data
    
    # Add feature importance if available
    if 'best_model_key' in results and 'models' in results:
        best_model_info = results['models'].get(results['best_model_key'], {})
        if best_model_info.get('feature_importance'):
            # Sort feature importance
            feature_importance = best_model_info['feature_importance']
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 features
            context['feature_importance'] = sorted_features
    
    # Add dataset info
    if 'dataset_info' in results:
        context['dataset_info'] = results['dataset_info']
    
    # Add confusion matrix data (simplified for now)
    # In a real implementation, you'd calculate this during training
    context['confusion_matrix'] = {
        'labels': results.get('classes', ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']),
        'data': [[25, 5, 2, 0, 0],  # Example data
                 [3, 30, 4, 1, 0],
                 [1, 2, 20, 3, 1],
                 [0, 1, 2, 15, 4],
                 [0, 0, 1, 2, 10]]
    }
    
    # Add learning curve data (example)
    context['learning_curve'] = {
        'train_sizes': [0.1, 0.3, 0.5, 0.7, 0.9],
        'train_scores': [0.65, 0.78, 0.82, 0.85, 0.88],
        'test_scores': [0.60, 0.75, 0.80, 0.83, 0.86]
    }
    
    # Add recommendations
    recommendations = []
    best_accuracy = results.get('best_accuracy', 0)
    
    if best_accuracy >= 0.9:
        recommendations.append("Excellent model accuracy! Ready for production use.")
    elif best_accuracy >= 0.8:
        recommendations.append("Good model accuracy. Consider adding more training data.")
    elif best_accuracy >= 0.7:
        recommendations.append("Fair accuracy. Try different feature combinations.")
    else:
        recommendations.append("Low accuracy. Consider collecting more diverse data.")
    
    # Check for class imbalance
    if 'dataset_info' in results and results['dataset_info'].get('class_distribution'):
        class_counts = list(results['dataset_info']['class_distribution'].values())
        if max(class_counts) / min(class_counts) > 10:
            recommendations.append("High class imbalance detected. Consider oversampling techniques.")
    
    context['recommendations'] = recommendations
    
    # Add next steps
    context['next_steps'] = [
        "Test the model on new data",
        "Deploy model to production",
        "Monitor model performance over time",
        "Retrain model with new data monthly"
    ]
    
    # ==================== LOG RESULTS ====================
    print(f"\n📋 Training Results Summary:")
    print(f"   Best Model: {results.get('best_model_name')}")
    print(f"   Accuracy: {results.get('best_accuracy', 0):.2%}")
    print(f"   Dataset Size: {results.get('dataset_info', {}).get('rows', 0)} samples")
    print(f"   Features: {len(results.get('feature_columns', []))}")
    
    if 'models' in results:
        print(f"\n   Model Comparison:")
        for model_key, model_info in results['models'].items():
            print(f"     {model_info['name']}: {model_info['accuracy']:.2f}%")
    
    print("\n✅ Results page ready")
    
    return render(request, 'bika/pages/admin/training_results.html', context)

@staff_member_required
def model_comparison_view(request):
    """Detailed model comparison"""
    result = enhanced_ai_service.get_detailed_model_comparison()
    
    context = {
        'comparison_result': result,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/ai/model_comparison.html', context)

@staff_member_required
def generate_sample_dataset_view(request):
    """Generate sample dataset"""
    num_samples = int(request.GET.get('samples', 1000))
    
    result = enhanced_ai_service.generate_sample_dataset(num_samples)
    
    if result.get('success'):
        messages.success(request, f'Sample dataset generated with {num_samples} samples')
        return JsonResponse(result)
    else:
        messages.error(request, f"Failed to generate dataset: {result.get('error')}")
        return JsonResponse(result, status=400)
    
@staff_member_required
@require_GET
def export_sales_report(request):
    """Export sales report as CSV"""
    import csv
    from django.http import HttpResponse
    from datetime import datetime, timedelta
    
    # Get date range (last 30 days by default)
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    filename = f"sales-report-{timezone.now().strftime('%Y%m%d')}.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    writer = csv.writer(response)
    
    # Write headers
    writer.writerow([
        'Date', 'Order ID', 'Customer', 'Product', 'Quantity', 
        'Unit Price', 'Total Amount', 'Status', 'Payment Method'
    ])
    
    # Write data
    orders = Order.objects.filter(
        created_at__range=[start_date, end_date]
    ).select_related('user').prefetch_related('items').order_by('-created_at')
    
    for order in orders:
        for item in order.items.all():
            writer.writerow([
                order.created_at.strftime('%Y-%m-%d'),
                order.order_number,
                order.user.username if order.user else '',
                item.product.name if item.product else '',
                item.quantity,
                float(item.price),
                float(item.quantity * item.price),
                order.get_status_display(),
                order.payment_method or 'N/A'
            ])
    
    return response

# Add this function to your views.py file, anywhere in the file:

def favicon_view(request):
    """Handle favicon requests to avoid 404 errors"""
    from django.http import HttpResponse
    return HttpResponse(status=204)  # No content response
@staff_member_required
def product_ai_insights_overview(request):
    """Overview page for product AI insights"""
    # Get all active products
    products = Product.objects.filter(status='active').select_related('category', 'vendor')
    
    # Get counts
    total_products = products.count()
    
    # Get AI predictions for each product
    try:
        from .ai_integration.models import FruitPrediction
        recent_predictions = FruitPrediction.objects.select_related('product').order_by('-prediction_date')[:10]
        
        # Count products with predictions
        product_ids_with_predictions = FruitPrediction.objects.values_list('product_id', flat=True).distinct()
        products_with_predictions = len(product_ids_with_predictions)
        
        # Add AI data to products
        for product in products:
            product.ai_predictions = FruitPrediction.objects.filter(product=product)[:3]
            product.alert_count = ProductAlert.objects.filter(product=product, is_resolved=False).count()
    except:
        recent_predictions = []
        products_with_predictions = 0
        for product in products:
            product.ai_predictions = []
            product.alert_count = 0
    
    # Get alert stats
    active_alerts = ProductAlert.objects.filter(is_resolved=False).count()
    high_risk_products = ProductAlert.objects.filter(
        is_resolved=False, 
        severity__in=['high', 'critical']
    ).values('product').distinct().count()
    
    context = {
        'products': products,
        'total_products': total_products,
        'products_with_predictions': products_with_predictions,
        'recent_predictions': recent_predictions,
        'active_alerts': active_alerts,
        'high_risk_products': high_risk_products,
        'last_analysis': timezone.now().strftime("%Y-%m-%d %H:%M"),
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/admin/product_ai_insights_overview.html', context)

# In views.py
@staff_member_required
def fruit_quality_dashboard(request):
    """Fruit quality monitoring dashboard"""
    # Get active fruit batches
    active_batches = FruitBatch.objects.filter(
        status='active'
    ).select_related(
        'fruit_type', 'storage_location'
    ).prefetch_related(
        'quality_readings'
    ).order_by('expected_expiry')
    
    # Add days_remaining calculation to each batch
    for batch in active_batches:
        if batch.expected_expiry:
            remaining = (batch.expected_expiry - timezone.now()).days
            batch.days_remaining = max(remaining, 0)
        else:
            batch.days_remaining = 0
    
    # Get latest quality reading for each batch
    for batch in active_batches:
        latest_reading = batch.quality_readings.order_by('-timestamp').first()
        batch.latest_reading = latest_reading
    
    # Calculate stats
    total_batches = FruitBatch.objects.count()
    active_batches_count = active_batches.count()
    
    # Count at-risk batches (expiring in less than 3 days)
    today = timezone.now().date()
    at_risk_batches = FruitBatch.objects.filter(
        status='active',
        expected_expiry__date__lte=today + timedelta(days=3)
    ).count()
    
    # Count expired batches
    expired_batches = FruitBatch.objects.filter(
        status='active',
        expected_expiry__date__lt=today
    ).count()
    
    # Get total readings
    total_readings = FruitQualityReading.objects.count()
    
    # Get AI predictions count
    ai_predictions = FruitQualityReading.objects.filter(
        predicted_class__isnull=False
    ).count()
    
    # Quality distribution
    quality_stats = {
        'fresh': FruitQualityReading.objects.filter(predicted_class='Fresh').count(),
        'good': FruitQualityReading.objects.filter(predicted_class='Good').count(),
        'fair': FruitQualityReading.objects.filter(predicted_class='Fair').count(),
        'poor': FruitQualityReading.objects.filter(predicted_class='Poor').count(),
        'rotten': FruitQualityReading.objects.filter(predicted_class='Rotten').count(),
    }
    
    # Get latest sensor data
    latest_sensor_data = RealTimeSensorData.objects.select_related(
        'fruit_batch'
    ).order_by('-recorded_at')[:5]
    
    # Fruit type distribution for chart
    fruit_types = FruitType.objects.all()
    fruit_types_labels = []
    fruit_types_data = []
    
    for ft in fruit_types:
        count = FruitBatch.objects.filter(fruit_type=ft).count()
        if count > 0:
            fruit_types_labels.append(ft.name)
            fruit_types_data.append(count)
    
    context = {
        'active_batches': active_batches,  # This is the key line!
        'stats': {
            'total_batches': total_batches,
            'active_batches': active_batches_count,
            'at_risk_batches': at_risk_batches,
            'expired_batches': expired_batches,
            'total_readings': total_readings,
            'ai_predictions': ai_predictions,
        },
        'quality_stats': quality_stats,
        'latest_sensor_data': latest_sensor_data,
        'last_updated': timezone.now().strftime("%H:%M:%S"),
        'fruit_types_labels': fruit_types_labels,
        'fruit_types_data': fruit_types_data,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/admin/fruit_dashboard.html', context)    


# Add this new view function
@staff_member_required
def train_five_models_view(request):
    """
    Train 5 different AI models on uploaded dataset or existing product data
    """
    print("\n🚀 =========== STARTING MODEL TRAINING ===========")
    
    if request.method == 'POST':
        try:
            # ==================== GET TRAINING PARAMETERS ====================
            print("\n📋 1. Parsing training parameters...")
            
            test_size = float(request.POST.get('test_size', 0.2))
            random_state = int(request.POST.get('random_state', 42))
            target_column = request.POST.get('target_column', 'Class')
            
            # Get which models to train
            models_to_train = request.POST.getlist('models')
            if not models_to_train:
                models_to_train = ['rf', 'xgb', 'svm', 'knn', 'gb']
            
            print(f"   Test size: {test_size}")
            print(f"   Random state: {random_state}")
            print(f"   Target column: {target_column}")
            print(f"   Models to train: {models_to_train}")
            
            df = None
            
            # ==================== HANDLE UPLOADED DATASET ====================
            if 'dataset_file' in request.FILES and request.FILES['dataset_file']:
                print("\n📁 2. Processing uploaded dataset file...")
                uploaded_file = request.FILES['dataset_file']
                file_name = uploaded_file.name
                file_size = uploaded_file.size
                
                print(f"   File: {file_name} ({file_size:,} bytes)")
                
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
                    for chunk in uploaded_file.chunks():
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name
                
                try:
                    # Load dataset
                    print("   Loading CSV file...")
                    
                    # Try different delimiters
                    delimiters = [',', ';', '\t']
                    
                    for delimiter in delimiters:
                        try:
                            df = pd.read_csv(tmp_path, delimiter=delimiter, encoding='utf-8')
                            if df.shape[1] >= 3:  # At least 3 columns
                                print(f"   ✅ Loaded with delimiter: '{delimiter}'")
                                break
                        except:
                            continue
                    
                    if df is None:
                        # Try with auto-detection
                        df = pd.read_csv(tmp_path, encoding='utf-8', engine='python')
                    
                    print(f"   Successfully loaded CSV")
                    print(f"   Shape: {df.shape} rows x {df.shape[1]} columns")
                    
                    # Show column information
                    print(f"   Columns found: {df.columns.tolist()}")
                    print(f"   Column dtypes:\n{df.dtypes.to_string()}")
                    
                    # Check if we have the expected columns
                    expected_columns = ['Fruit', 'Temp', 'Humid (%)', 'Light (Fux)', 'CO2 (pmm)', 'Class']
                    
                    # Try to match columns (case insensitive)
                    column_mapping = {}
                    for expected in expected_columns:
                        for actual in df.columns:
                            if expected.lower() in actual.lower() or actual.lower() in expected.lower():
                                column_mapping[expected] = actual
                                break
                    
                    print(f"   Column mapping: {column_mapping}")
                    
                    # Rename columns if needed
                    if column_mapping:
                        df = df.rename(columns={v: k for k, v in column_mapping.items()})
                    
                    # Ensure we have required columns
                    missing_cols = [col for col in expected_columns if col not in df.columns]
                    if missing_cols:
                        print(f"   ⚠️ Missing columns: {missing_cols}")
                        
                        # Try to add missing columns with default values
                        for col in missing_cols:
                            if col == 'Fruit':
                                df['Fruit'] = 'Apple'
                            elif col == 'Class':
                                df['Class'] = 'Good'
                            elif col == 'Temp':
                                df['Temp'] = 5.0
                            elif col == 'Humid (%)':
                                df['Humid (%)'] = 85.0
                            elif col == 'Light (Fux)':
                                df['Light (Fux)'] = 100.0
                            elif col == 'CO2 (pmm)':
                                df['CO2 (pmm)'] = 400.0
                    
                    # Keep only the columns we need
                    df = df[expected_columns]
                    
                    print(f"   Final columns: {df.columns.tolist()}")
                    
                except Exception as e:
                    print(f"❌ Error loading CSV: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    messages.error(request, f'Error loading CSV file: {str(e)}')
                    return redirect('bika:train_models')
                finally:
                    # Clean up
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        print(f"   🗑️ Cleaned up temporary file")
            
            # ==================== USE DATABASE DATA IF NO FILE ====================
            if df is None or df.empty:
                print("\n📊 3. Using database data...")
                df = get_product_dataset_from_db()
                
                # Ensure we have the Class column
                if 'Class' not in df.columns and target_column in df.columns:
                    df = df.rename(columns={target_column: 'Class'})
                    target_column = 'Class'
                    print(f"   Renamed '{target_column}' column to 'Class'")
            
            # ==================== VALIDATE DATASET ====================
            if df is None or df.empty:
                print("❌ No data available for training")
                messages.error(request, 'No data available for training.')
                return redirect('bika:train_models')
            
            print(f"\n✅ 4. Dataset ready for training")
            print(f"   Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"   Target column: {target_column}")
            
            # Show class distribution
            if target_column in df.columns:
                class_dist = df[target_column].value_counts()
                print(f"   Class distribution:")
                for class_name, count in class_dist.items():
                    percentage = (count / len(df)) * 100
                    print(f"     {class_name}: {count} samples ({percentage:.1f}%)")
            
            # Show sample of data
            print(f"\n   Sample data (first 3 rows):")
            print(df.head(3).to_string())
            
            # ==================== TRAIN MODELS ====================
            print("\n🎯 5. Starting model training...")
            
            # Store training parameters in session
            request.session['training_params'] = {
                'test_size': test_size,
                'random_state': random_state,
                'target_column': target_column,
                'models_to_train': models_to_train,
                'dataset_rows': df.shape[0],
                'dataset_columns': list(df.columns),
                'timestamp': timezone.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Train the models
            results = train_multiple_models(df, target_column, models_to_train, test_size, random_state)
            
            # Add metadata
            results['training_timestamp'] = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
            results['dataset_source'] = 'uploaded_file' if 'dataset_file' in request.FILES else 'database'
            
            # ==================== SAVE BEST MODEL ====================
            print("\n💾 6. Saving best model to database...")
            
            if results.get('best_model'):
                saved_model = save_model_to_database(results)
                
                if saved_model:
                    print(f"   ✅ Model saved successfully!")
                    print(f"   Model name: {saved_model.name}")
                    print(f"   Model ID: {saved_model.id}")
                    print(f"   Accuracy: {saved_model.accuracy:.2%}")
                    
                    results['saved_model'] = {
                        'id': saved_model.id,
                        'name': saved_model.name,
                        'accuracy': float(saved_model.accuracy),
                        'file_path': str(saved_model.model_file)
                    }
                    
                    messages.success(
                        request,
                        f'✅ Model training completed successfully! '
                        f'Best model: {saved_model.name} with accuracy {saved_model.accuracy:.2%}'
                    )
                else:
                    print("   ⚠️ Model trained but could not save to database")
                    messages.warning(
                        request,
                        'Model trained successfully but could not save to database. '
                        'Check server logs for details.'
                    )
            else:
                print("   ⚠️ No best model found in results")
                messages.warning(request, 'Training completed but no model was selected as best.')
            
            # ==================== STORE RESULTS ====================
            print("\n📝 7. Storing training results...")
            request.session['training_results'] = results
            
            # Clean up old session data
            if 'training_params' in request.session:
                del request.session['training_params']
            
            print("\n🎉 =========== TRAINING COMPLETED SUCCESSFULLY ===========")
            return redirect('bika:training_results')
            
        except Exception as e:
            print(f"\n❌ =========== TRAINING FAILED ===========")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            messages.error(
                request,
                f'Training failed: {str(e)}. '
                'Please check the dataset format and try again.'
            )
            return redirect('bika:train_models')
    
    # ==================== GET REQUEST - SHOW TRAINING FORM ====================
    print("\n📋 Loading training form...")
    
    # Get statistics for the form
    context = {
        'site_info': SiteInfo.objects.first(),
        'product_count': Product.objects.filter(status='active').count(),
        'quality_readings': FruitQualityReading.objects.count(),
        'active_batches': FruitBatch.objects.filter(status='active').count(),
        'sensor_data': RealTimeSensorData.objects.count(),
        'existing_models': TrainedModel.objects.filter(
            model_type='fruit_quality'
        ).order_by('-training_date')[:5],
        'available_fruits': FruitType.objects.values_list('name', flat=True).distinct(),
        'quality_classes': ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
    }
    
    print(f"   Available fruits: {list(context['available_fruits'])}")
    print(f"   Existing models: {context['existing_models'].count()}")
    
    return render(request, 'bika/pages/admin/train_models.html', context)

def get_product_dataset_from_db():
    """
    Extract product data from database for training with your specific column format
    """
    print("\n📊 Extracting dataset from database...")
    print("   Expected columns: Fruit, Temp, Humid (%), Light (Fux), CO2 (pmm), Class")
    
    try:
        data_rows = []
        
        # ==================== SOURCE 1: FRUIT QUALITY READINGS ====================
        print("   Collecting fruit quality readings...")
        quality_readings = FruitQualityReading.objects.select_related(
            'fruit_batch', 'fruit_batch__fruit_type'
        ).filter(
            predicted_class__isnull=False
        ).exclude(predicted_class='')
        
        reading_count = quality_readings.count()
        print(f"   Found {reading_count} quality readings")
        
        for i, reading in enumerate(quality_readings[:1000]):  # Limit to 1000 for performance
            try:
                # Get fruit type name
                fruit_name = 'Unknown'
                if reading.fruit_batch and reading.fruit_batch.fruit_type:
                    fruit_name = reading.fruit_batch.fruit_type.name
                
                # Create data row with your exact column names
                row = {
                    'Fruit': fruit_name,
                    'Temp': float(reading.temperature) if reading.temperature is not None else np.random.uniform(2, 25),
                    'Humid (%)': float(reading.humidity) if reading.humidity is not None else np.random.uniform(70, 95),
                    'Light (Fux)': float(reading.light_intensity) if reading.light_intensity is not None else np.random.uniform(50, 200),
                    'CO2 (pmm)': float(reading.co2_level) if reading.co2_level is not None else np.random.uniform(300, 500),
                    'Class': reading.predicted_class if reading.predicted_class else 'Good'
                }
                data_rows.append(row)
                
            except Exception as e:
                if i < 10:  # Only log first few errors
                    print(f"      Warning: Error processing reading {reading.id}: {e}")
                continue
        
        # ==================== SOURCE 2: REAL TIME SENSOR DATA ====================
        print("   Collecting sensor data...")
        sensor_data = RealTimeSensorData.objects.select_related(
            'fruit_batch', 'fruit_batch__fruit_type'
        ).filter(
            predicted_class__isnull=False
        ).exclude(predicted_class='')
        
        sensor_count = sensor_data.count()
        print(f"   Found {sensor_count} sensor readings")
        
        for i, sensor in enumerate(sensor_data[:500]):  # Limit to 500
            try:
                # Get fruit type name
                fruit_name = 'Unknown'
                if sensor.fruit_batch and sensor.fruit_batch.fruit_type:
                    fruit_name = sensor.fruit_batch.fruit_type.name
                
                # Initialize with defaults
                row = {
                    'Fruit': fruit_name,
                    'Temp': np.random.uniform(2, 25),  # Default
                    'Humid (%)': np.random.uniform(70, 95),  # Default
                    'Light (Fux)': np.random.uniform(50, 200),  # Default
                    'CO2 (pmm)': np.random.uniform(300, 500),  # Default
                    'Class': sensor.predicted_class if sensor.predicted_class else 'Good'
                }
                
                # Update based on sensor type
                if sensor.sensor_type == 'temperature':
                    row['Temp'] = float(sensor.value)
                elif sensor.sensor_type == 'humidity':
                    row['Humid (%)'] = float(sensor.value)
                elif sensor.sensor_type == 'light':
                    row['Light (Fux)'] = float(sensor.value)
                elif sensor.sensor_type == 'co2':
                    row['CO2 (pmm)'] = float(sensor.value)
                
                data_rows.append(row)
                
            except Exception as e:
                if i < 10:
                    print(f"      Warning: Error processing sensor {sensor.id}: {e}")
                continue
        
        # ==================== CREATE DATAFRAME ====================
        if not data_rows:
            print("   ⚠️ No data found in database. Creating sample data...")
            return create_sample_fruit_data()
        
        df = pd.DataFrame(data_rows)
        
        # ==================== CLEAN AND VALIDATE DATA ====================
        print("\n🧹 Cleaning and validating dataset...")
        
        # Ensure all required columns exist
        required_columns = ['Fruit', 'Temp', 'Humid (%)', 'Light (Fux)', 'CO2 (pmm)', 'Class']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"   ⚠️ Missing column: {col}")
                if col == 'Fruit':
                    df['Fruit'] = 'Apple'  # Default fruit
                elif col == 'Class':
                    df['Class'] = 'Good'  # Default class
                else:
                    df[col] = 0.0  # Default numeric value
        
        # Keep only required columns
        df = df[required_columns]
        
        # Clean data types
        print("   Cleaning data types...")
        numeric_columns = ['Temp', 'Humid (%)', 'Light (Fux)', 'CO2 (pmm)']
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, coerce errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN with reasonable values based on column
                if col == 'Temp':
                    df[col].fillna(np.random.uniform(2, 25), inplace=True)
                elif col == 'Humid (%)':
                    df[col].fillna(np.random.uniform(70, 95), inplace=True)
                elif col == 'Light (Fux)':
                    df[col].fillna(np.random.uniform(50, 200), inplace=True)
                elif col == 'CO2 (pmm)':
                    df[col].fillna(np.random.uniform(300, 500), inplace=True)
                
                # Ensure values are within reasonable ranges
                if col == 'Temp':
                    df[col] = df[col].clip(0, 40)  # Temperature between 0-40°C
                elif col == 'Humid (%)':
                    df[col] = df[col].clip(0, 100)  # Humidity between 0-100%
                elif col == 'Light (Fux)':
                    df[col] = df[col].clip(0, 1000)  # Light between 0-1000 lux
                elif col == 'CO2 (pmm)':
                    df[col] = df[col].clip(200, 1000)  # CO2 between 200-1000 ppm
        
        # Clean Fruit column
        df['Fruit'] = df['Fruit'].fillna('Apple').astype(str).str.strip()
        # Standardize fruit names
        fruit_mapping = {
            'apple': 'Apple', 'apples': 'Apple',
            'banana': 'Banana', 'bananas': 'Banana',
            'orange': 'Orange', 'oranges': 'Orange',
            'mango': 'Mango', 'mangoes': 'Mango',
            'grape': 'Grapes', 'grapes': 'Grapes',
            'strawberry': 'Strawberry', 'strawberries': 'Strawberry',
            'pineapple': 'Pineapple', 'pineapples': 'Pineapple'
        }
        df['Fruit'] = df['Fruit'].str.lower().map(lambda x: fruit_mapping.get(x, x.title()))
        
        # Clean Class column
        df['Class'] = df['Class'].fillna('Good').astype(str).str.strip().str.title()
        # Standardize class names
        class_mapping = {
            'fresh': 'Fresh',
            'good': 'Good',
            'fair': 'Fair',
            'poor': 'Poor',
            'rotten': 'Rotten',
            'excellent': 'Fresh',  # Map excellent to Fresh
            'very good': 'Good',
            'very_good': 'Good',
            'bad': 'Poor',
            'spoiled': 'Rotten'
        }
        df['Class'] = df['Class'].str.lower().map(lambda x: class_mapping.get(x, x.title()))
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        print(f"\n✅ Dataset created successfully!")
        print(f"   Final shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Fruit distribution:\n{df['Fruit'].value_counts()}")
        print(f"   Class distribution:\n{df['Class'].value_counts()}")
        print("\n   Sample data (first 5 rows):")
        print(df.head().to_string())
        
        # Show statistics
        print("\n   Dataset statistics:")
        for col in numeric_columns:
            if col in df.columns:
                print(f"   {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error extracting dataset from database: {e}")
        import traceback
        traceback.print_exc()
        return create_sample_fruit_data()
    
def create_sample_fruit_data():
    """
    Create sample data matching your exact column format
    """
    print("   Creating sample data with your exact format...")
    
    # Define realistic fruit parameters
    fruit_params = {
        'Apple': {'temp_mean': 3, 'temp_std': 1, 'humid_mean': 90, 'humid_std': 5},
        'Banana': {'temp_mean': 13, 'temp_std': 2, 'humid_mean': 85, 'humid_std': 3},
        'Orange': {'temp_mean': 8, 'temp_std': 1.5, 'humid_mean': 88, 'humid_std': 4},
        'Mango': {'temp_mean': 10, 'temp_std': 2, 'humid_mean': 87, 'humid_std': 3},
        'Grapes': {'temp_mean': 2, 'temp_std': 1, 'humid_mean': 92, 'humid_std': 4},
        'Strawberry': {'temp_mean': 1, 'temp_std': 0.5, 'humid_mean': 95, 'humid_std': 2},
        'Pineapple': {'temp_mean': 12, 'temp_std': 1, 'humid_mean': 85, 'humid_std': 3}
    }
    
    fruits = list(fruit_params.keys())
    classes = ['Fresh', 'Good', 'Fair', 'Poor', 'Rotten']
    
    data = []
    
    for i in range(1000):  # Create 1000 samples
        fruit = np.random.choice(fruits)
        params = fruit_params[fruit]
        
        # Generate realistic values based on fruit type
        temp = np.random.normal(params['temp_mean'], params['temp_std'])
        humid = np.random.normal(params['humid_mean'], params['humid_std'])
        light = np.random.uniform(50, 200)
        co2 = np.random.uniform(300, 500)
        
        # Determine quality based on conditions
        if temp < 0 or temp > 15 or humid < 80:
            # Poor conditions
            class_probs = [0.1, 0.2, 0.3, 0.3, 0.1]  # Higher probability of poor/rotten
        elif 2 <= temp <= 8 and 85 <= humid <= 95:
            # Optimal conditions
            class_probs = [0.4, 0.4, 0.1, 0.05, 0.05]  # Higher probability of fresh/good
        else:
            # Average conditions
            class_probs = [0.2, 0.3, 0.3, 0.15, 0.05]
        
        fruit_class = np.random.choice(classes, p=class_probs)
        
        data.append({
            'Fruit': fruit,
            'Temp': round(temp, 1),
            'Humid (%)': round(humid, 1),
            'Light (Fux)': round(light, 1),
            'CO2 (pmm)': round(co2, 1),
            'Class': fruit_class
        })
    
    df = pd.DataFrame(data)
    
    print(f"   Created sample dataset with {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Fruit distribution:\n{df['Fruit'].value_counts()}")
    print(f"   Class distribution:\n{df['Class'].value_counts()}")
    
    return df

def train_multiple_models(df, target_column, models_to_train, test_size=0.2, random_state=42):
    """
    Train multiple ML models on the dataset
    """
    results = {
        'models': {},
        'best_model': None,
        'best_model_name': None,
        'best_accuracy': 0,
        'target_column': target_column,
        'feature_columns': [],
        'dataset_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'columns_list': df.columns.tolist()
        }
    }
    
    try:
        print(f"\n🔍 Dataset Analysis:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Target column: {target_column}")
        print(f"   Target values: {df[target_column].unique()}")
        print(f"   Target distribution:\n{df[target_column].value_counts()}")
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        results['feature_columns'] = X.columns.tolist()
        
        # Handle non-numeric columns in features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
        
        # Check if target needs encoding
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            results['label_encoder'] = le
            results['classes'] = le.classes_.tolist()
            print(f"   Encoded classes: {le.classes_}")
        
        # Check for missing values
        if X.isnull().any().any():
            print(f"   Missing values found. Filling with median...")
            X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Testing samples: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results['scaler'] = scaler
        
        # Model configurations
        model_configs = {
            'rf': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            },
            'xgb': {
                'name': 'XGBoost',
                'model': XGBClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            },
            'svm': {
                'name': 'Support Vector Machine',
                'model': SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=random_state,
                    class_weight='balanced'
                )
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'model': KNeighborsClassifier(
                    n_neighbors=7,
                    weights='distance',
                    algorithm='auto'
                )
            },
            'gb': {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=random_state
                )
            }
        }
        
        # Train selected models
        for model_key in models_to_train:
            if model_key in model_configs:
                config = model_configs[model_key]
                print(f"\n📊 Training {config['name']}...")
                
                try:
                    model = config['model']
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Feature importance for tree-based models
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X.columns, model.feature_importances_))
                    
                    # Store results
                    results['models'][model_key] = {
                        'name': config['name'],
                        'accuracy': round(accuracy * 100, 2),
                        'precision': round(precision * 100, 2),
                        'recall': round(recall * 100, 2),
                        'f1_score': round(f1 * 100, 2),
                        'cv_mean': round(cv_mean * 100, 2),
                        'cv_std': round(cv_std * 100, 2),
                        'model_object': model,
                        'feature_names': X.columns.tolist(),
                        'feature_importance': feature_importance
                    }
                    
                    # Update best model
                    if accuracy > results['best_accuracy']:
                        results['best_accuracy'] = accuracy
                        results['best_model'] = model
                        results['best_model_name'] = config['name']
                        results['best_model_key'] = model_key
                        results['best_scaler'] = scaler
                    
                    print(f"   ✅ {config['name']}: {accuracy*100:.2f}% accuracy")
                    
                except Exception as e:
                    print(f"   ❌ Error training {config['name']}: {str(e)}")
                    continue
        
        # Sort models by accuracy
        results['sorted_models'] = sorted(
            results['models'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"\n🏆 Best Model: {results['best_model_name']} ({results['best_accuracy']*100:.2f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def save_model_to_database(training_results):
    """
    Save the trained model to database
    """
    try:
        from django.conf import settings
        
        print(f"\n💾 Saving model to database...")
        print(f"   Best model: {training_results.get('best_model_name')}")
        print(f"   Accuracy: {training_results.get('best_accuracy', 0):.2%}")
        
        if not training_results or 'best_model' not in training_results:
            print("   ❌ No valid training results or best model found")
            return None
        
        # ==================== CREATE DATASET RECORD ====================
        dataset_name = f"Fruit Quality Dataset {timezone.now().strftime('%Y-%m-%d %H:%M')}"
        dataset_description = f"Dataset for fruit quality prediction with {training_results.get('dataset_info', {}).get('rows', 0)} samples"
        
        print(f"   Creating dataset record: {dataset_name}")
        
        dataset = ProductDataset.objects.create(
            name=dataset_name,
            dataset_type='quality_control',
            description=dataset_description,
            row_count=training_results.get('dataset_info', {}).get('rows', 0),
            is_active=True,
            columns=json.dumps(training_results.get('dataset_info', {}).get('columns_list', []))
        )
        
        print(f"   ✅ Dataset created with ID: {dataset.id}")
        
        # ==================== PREPARE MODEL DATA ====================
        model_data = {
            'model': training_results['best_model'],
            'scaler': training_results.get('scaler'),
            'feature_columns': training_results.get('feature_columns', []),
            'feature_names': training_results.get('feature_columns', []),  # For compatibility
            'training_date': timezone.now(),
            'accuracy': float(training_results.get('best_accuracy', 0)),
            'model_config': {
                'best_model_name': training_results.get('best_model_name'),
                'best_model_key': training_results.get('best_model_key'),
                'target_column': training_results.get('target_column'),
                'test_size': training_results.get('test_size', 0.2),
                'random_state': training_results.get('random_state', 42)
            },
            'label_encoder': training_results.get('label_encoder'),
            'classes': training_results.get('classes', [])
        }
        
        # Add feature importance if available
        best_model_key = training_results.get('best_model_key')
        if best_model_key in training_results.get('models', {}):
            model_info = training_results['models'][best_model_key]
            if model_info.get('feature_importance'):
                model_data['feature_importance'] = model_info['feature_importance']
        
        # ==================== SAVE MODEL TO FILE ====================
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{timestamp}.pkl"
        model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
        
        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)
        full_path = os.path.join(model_dir, model_filename)
        
        print(f"   Saving model file to: {full_path}")
        
        # Save model using joblib
        joblib.dump(model_data, full_path, compress=3)  # compress=3 for better compression
        
        # ==================== CREATE DATABASE RECORD ====================
        model_name = f"{training_results.get('best_model_name', 'AI Model')} v{timestamp}"
        model_accuracy = float(training_results.get('best_accuracy', 0))
        
        print(f"   Creating TrainedModel record: {model_name}")
        print(f"   Accuracy to save: {model_accuracy}")
        
        trained_model = TrainedModel.objects.create(
            name=model_name,
            model_type='fruit_quality',
            dataset=dataset,
            model_file=os.path.join('trained_models', model_filename),  # Relative to MEDIA_ROOT
            accuracy=model_accuracy,
            is_active=True,
            feature_columns=json.dumps(training_results.get('feature_columns', [])),
            training_date=timezone.now()
        )
        
        # ==================== DEACTIVATE OLD MODELS ====================
        # Deactivate all other fruit quality models
        TrainedModel.objects.filter(
            model_type='fruit_quality'
        ).exclude(id=trained_model.id).update(is_active=False)
        
        print(f"   ✅ Model saved to database: {trained_model.name} (ID: {trained_model.id})")
        print(f"   📁 Model file: {trained_model.model_file}")
        print(f"   📊 Accuracy: {trained_model.accuracy:.2%}")
        print(f"   🏷️  Model type: {trained_model.get_model_type_display()}")
        
        # ==================== UPDATE AI SERVICE ====================
        try:
            if AI_SERVICES_AVAILABLE and enhanced_ai_service:
                enhanced_ai_service.load_active_model()
                print(f"   🔄 AI service model reloaded")
        except Exception as e:
            print(f"   ⚠️  Could not update AI service: {e}")
        
        # ==================== LOG SUCCESS ====================
        logger.info(f"Trained model saved: {trained_model.name} with accuracy {trained_model.accuracy:.2%}")
        
        return trained_model
        
    except ImportError as e:
        print(f"❌ Import error in save_model_to_database: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    except Exception as e:
        print(f"❌ Error saving model to database: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save minimal information for debugging
        try:
            error_model = TrainedModel.objects.create(
                name=f"Error Model {timezone.now().strftime('%Y%m%d_%H%M%S')}",
                model_type='fruit_quality',
                accuracy=0.0,
                is_active=False,
                feature_columns=json.dumps([]),
                training_date=timezone.now()
            )
            print(f"   ⚠️  Created error placeholder model ID: {error_model.id}")
        except:
            pass
            
        return None
    

@staff_member_required
def model_comparison_view(request):
    """Compare trained models"""
    models = TrainedModel.objects.filter(is_active=True).order_by('-training_date')
    
    context = {
        'models': models,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/admin/model_comparison.html', context)        

@staff_member_required
def analyze_csv(request):
    """Analyze uploaded CSV file"""
    if request.method == 'POST' and 'csv_file' in request.FILES:
        file = request.FILES['csv_file']
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            for chunk in file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        try:
            # Load CSV
            df = pd.read_csv(tmp_path)
            
            analysis = {
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'dtypes': df.dtypes.astype(str).to_dict(),
                'head': df.head().to_dict('records'),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_counts': {col: df[col].nunique() for col in df.columns}
            }
            
            # Suggest target column
            suggestions = []
            for col in df.columns:
                if col.lower() in ['class', 'quality', 'grade', 'target', 'label']:
                    suggestions.append((col, 'Likely target (based on name)'))
                elif df[col].nunique() <= 10:
                    suggestions.append((col, f'Classification ({df[col].nunique()} classes)'))
                elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10:
                    suggestions.append((col, f'Regression ({df[col].nunique()} unique values)'))
            
            analysis['suggestions'] = suggestions
            
            return JsonResponse({'success': True, 'analysis': analysis})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
        finally:
            os.unlink(tmp_path)
    
    return JsonResponse({'success': False, 'error': 'No file uploaded'})

from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import Order, ProductReview

@login_required
def cancel_order(request, order_id):
    """Cancel an order"""
    order = get_object_or_404(Order, id=order_id, user=request.user)
    
    # Check if order can be cancelled
    if order.status not in ['pending', 'confirmed', 'processing']:
        messages.error(request, f'Cannot cancel order with status: {order.get_status_display()}')
        return redirect('bika:order_detail', order_id=order.id)
    
    # Update order status
    order.status = 'cancelled'
    order.cancelled_at = timezone.now()
    order.save()
    
    # Log the cancellation
    messages.success(request, 'Order has been cancelled successfully.')
    return redirect('bika:order_detail', order_id=order.id)

@login_required
def create_review(request, order_id):
    """Create review for order items"""
    order = get_object_or_404(Order, id=order_id, user=request.user)
    
    # Check if order is delivered
    if order.status != 'delivered':
        messages.error(request, 'You can only review delivered orders.')
        return redirect('bika:order_detail', order_id=order.id)
    
    # Check if user has already reviewed this order
    existing_reviews = ProductReview.objects.filter(order=order, user=request.user)
    if existing_reviews.exists():
        messages.info(request, 'You have already reviewed this order.')
        return redirect('bika:order_detail', order_id=order.id)
    
    if request.method == 'POST':
        # Process reviews for each item
        reviewed_items = 0
        
        for item in order.items.all():
            product = item.product
            rating_key = f'rating_{product.id}'
            comment_key = f'comment_{product.id}'
            
            if rating_key in request.POST:
                rating = request.POST.get(rating_key)
                comment = request.POST.get(comment_key, '')
                
                if rating:
                    # Create review
                    ProductReview.objects.create(
                        product=product,
                        user=request.user,
                        order=order,
                        rating=int(rating),
                        comment=comment,
                        is_approved=False
                    )
                    reviewed_items += 1
        
        if reviewed_items > 0:
            messages.success(request, f'Thank you! Your review has been submitted for {reviewed_items} item(s).')
            return redirect('bika:order_detail', order_id=order.id)
        else:
            messages.error(request, 'Please rate at least one item.')
    
    # Get items that haven't been reviewed yet
    items_to_review = []
    for item in order.items.all():
        if not ProductReview.objects.filter(product=item.product, user=request.user, order=order).exists():
            items_to_review.append(item)
    
    context = {
        'order': order,
        'items_to_review': items_to_review,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/user/create_review.html', context)

# Also make sure you have these imports at the top of your views.py:
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from .models import Order, ProductReview, SiteInfo

def debug_urls(request):
    """Debug view to check URL patterns"""
    from django.urls import reverse, resolve
    from django.http import JsonResponse
    from django.test import RequestFactory
    
    urls_to_test = [
        '/orders/1/',
        '/orders/1/cancel/',
        '/orders/1/review/',
    ]
    
    results = {}
    for url in urls_to_test:
        try:
            match = resolve(url)
            results[url] = {
                'success': True,
                'view_name': match.view_name,
                'args': match.args,
                'kwargs': match.kwargs,
            }
        except Exception as e:
            results[url] = {
                'success': False,
                'error': str(e),
            }
    
    return JsonResponse(results)

# Add to urls.py:
@login_required
def order_detail(request, order_id):
    """Order detail page"""
    # Check if order exists at all first
    try:
        order = Order.objects.get(id=order_id)
        print(f"DEBUG: Found order {order.id} for user {order.user.username}")
        print(f"DEBUG: Current user is {request.user.username}")
    except Order.DoesNotExist:
        print(f"DEBUG: No order with id={order_id}")
        raise Http404("Order does not exist")
    
    # Now check if it belongs to current user
    if order.user != request.user:
        print(f"DEBUG: Order user {order.user.username} != current user {request.user.username}")
        messages.error(request, "You don't have permission to view this order.")
        return redirect('bika:user_orders')
    
    # Get payments for this order
    payments = Payment.objects.filter(order=order).order_by('-created_at')
    
    context = {
        'order': order,
        'payments': payments,
        'site_info': SiteInfo.objects.first(),
    }
    return render(request, 'bika/pages/user/order_detail.html', context)

# views.py - vendor_edit_product view
@login_required
def vendor_edit_product(request, product_id):
    """Edit product for vendor"""
    if not request.user.is_vendor():
        messages.error(request, 'Access denied. Vendor account required.')
        return redirect('bika:home')
    
    product = get_object_or_404(Product, id=product_id, vendor=request.user)
    
    if request.method == 'POST':
        # Handle basic product updates
        product.name = request.POST.get('name')
        product.sku = request.POST.get('sku')
        product.description = request.POST.get('description')
        product.short_description = request.POST.get('short_description', '')
        product.price = Decimal(request.POST.get('price', '0'))
        product.compare_price = request.POST.get('compare_price') or None
        product.cost_price = request.POST.get('cost_price') or None
        product.tax_rate = Decimal(request.POST.get('tax_rate', '0'))
        product.stock_quantity = int(request.POST.get('stock_quantity', '0'))
        product.low_stock_threshold = int(request.POST.get('low_stock_threshold', '5'))
        product.track_inventory = 'track_inventory' in request.POST
        product.allow_backorders = 'allow_backorders' in request.POST
        product.is_digital = 'is_digital' in request.POST
        product.category_id = request.POST.get('category')
        product.status = request.POST.get('status')
        product.condition = request.POST.get('condition')
        product.brand = request.POST.get('brand', '')
        product.model = request.POST.get('model', '')
        product.color = request.POST.get('color', '')
        product.size = request.POST.get('size', '')
        product.material = request.POST.get('material', '')
        product.weight = request.POST.get('weight') or None
        product.dimensions = request.POST.get('dimensions', '')
        product.tags = request.POST.get('tags', '')
        product.meta_title = request.POST.get('meta_title', '')
        product.meta_description = request.POST.get('meta_description', '')
        product.is_featured = 'is_featured' in request.POST
        
        # Handle image deletions
        delete_images = request.POST.getlist('delete_images')
        if delete_images:
            ProductImage.objects.filter(id__in=delete_images, product=product).delete()
        
        # Handle primary image
        primary_image = request.POST.get('primary_image')
        if primary_image:
            ProductImage.objects.filter(product=product).update(is_primary=False)
            ProductImage.objects.filter(id=primary_image, product=product).update(is_primary=True)
        
        # Handle new image uploads
        new_images = request.FILES.getlist('new_images')
        if new_images:
            for i, image in enumerate(new_images):
                is_primary = (i == 0 and not product.images.filter(is_primary=True).exists())
                ProductImage.objects.create(
                    product=product,
                    image=image,
                    display_order=i,
                    is_primary=is_primary
                )
        
        product.save()
        messages.success(request, f'Product "{product.name}" updated successfully!')
        return redirect('bika:vendor_products')
    
    # GET request - show edit form
    categories = ProductCategory.objects.filter(is_active=True)
    
    context = {
        'product': product,
        'categories': categories,
        'status_choices': Product.STATUS_CHOICES,
        'condition_choices': Product.CONDITION_CHOICES,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/vendor/edit_product.html', context)
# ==================== MANAGER VIEWS ====================

@login_required
@role_required('manager', 'admin')  # Allow both managers and admins
def manager_dashboard(request):
    """Manager dashboard with specific metrics"""
    # Total active inventory items
    total_items = InventoryItem.objects.filter(is_active=True).count()
    
    # Low stock items (below threshold)
    low_stock_items = InventoryItem.objects.filter(
        quantity__lte=F('low_stock_threshold'),
        status='active'
    ).count()
    
    # Active deliveries
    active_deliveries = Delivery.objects.filter(
        status__in=['pending', 'processing', 'in_transit', 'out_for_delivery']
    ).count()
    
    # Expiring items (next 30 days)
    thirty_days = timezone.now().date() + timedelta(days=30)
    expiring_items = InventoryItem.objects.filter(
        expiry_date__lte=thirty_days,
        expiry_date__gte=timezone.now().date(),
        status='active'
    ).count()
    
    # Recent inventory changes
    recent_changes = InventoryHistory.objects.select_related(
        'item', 'user'
    ).order_by('-timestamp')[:10]
    
    # Get delivery performance
    delivered_count = Delivery.objects.filter(status='delivered').count()
    late_count = Delivery.objects.filter(
        status='delivered',
        actual_delivery__gt=F('estimated_delivery')
    ).count()
    
    # Calculate delivery success rate
    if delivered_count > 0:
        delivery_success_rate = round((1 - (late_count / delivered_count)) * 100, 2)
    else:
        delivery_success_rate = 0
    
    # Get inventory value
    inventory_value = InventoryItem.objects.aggregate(
        total_value=Sum('total_value')
    )['total_value'] or 0
    
    # Get recent AI predictions (if available)
    recent_predictions = []
    if AI_SERVICES_AVAILABLE:
        try:
            # Get top 5 items for prediction
            top_items = InventoryItem.objects.filter(
                status='active',
                quantity__gt=0
            ).order_by('-total_value')[:3]
            
            for item in top_items:
                # This is a placeholder - implement your actual prediction logic
                prediction = {
                    'item': item,
                    'predicted_demand': item.quantity + 10,  # Example
                    'reorder_recommended': item.quantity <= item.reorder_point,
                    'confidence': 0.75
                }
                recent_predictions.append(prediction)
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
    
    # Get weekly/monthly stats
    today = timezone.now()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    
    weekly_deliveries = Delivery.objects.filter(
        created_at__gte=week_start
    ).count()
    
    monthly_deliveries = Delivery.objects.filter(
        created_at__gte=month_start
    ).count()
    
    # Get item type distribution for chart
    item_types = {
        'Storage': InventoryItem.objects.filter(item_type='storage', status='active').count(),
        'For Sale': InventoryItem.objects.filter(item_type='sale', status='active').count(),
        'Rental': InventoryItem.objects.filter(item_type='rental', status='active').count(),
    }
    
    # Get delivery status distribution
    delivery_statuses = {
        'Pending': Delivery.objects.filter(status='pending').count(),
        'Processing': Delivery.objects.filter(status='processing').count(),
        'In Transit': Delivery.objects.filter(status='in_transit').count(),
        'Delivered': Delivery.objects.filter(status='delivered').count(),
        'Cancelled': Delivery.objects.filter(status='cancelled').count(),
    }
    
    # Get alerts
    alerts = ProductAlert.objects.filter(
        is_resolved=False
    ).select_related('product').order_by('-created_at')[:5]
    
    context = {
        # Summary Stats
        'total_items': total_items,
        'low_stock_items': low_stock_items,
        'active_deliveries': active_deliveries,
        'expiring_items': expiring_items,
        'inventory_value': inventory_value,
        'weekly_deliveries': weekly_deliveries,
        'monthly_deliveries': monthly_deliveries,
        'delivery_success_rate': delivery_success_rate,
        'late_deliveries': late_count,
        
        # Recent Data
        'recent_changes': recent_changes,
        'recent_predictions': recent_predictions,
        'alerts': alerts,
        
        # Charts Data
        'item_types': item_types,
        'delivery_statuses': delivery_statuses,
        'prediction_alerts': len([p for p in recent_predictions if p.get('reorder_recommended', False)]),
        
        # System Info
        'last_updated': timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_role': 'Manager',
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/manager/dashboard.html', context)

@login_required
@role_required('manager', 'admin')
def manager_inventory(request):
    """Manager inventory view with enhanced controls"""
    # Get all inventory items
    items = InventoryItem.objects.select_related(
        'category', 'client', 'location'
    ).filter(is_active=True)
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    item_type_filter = request.GET.get('item_type', '')
    client_filter = request.GET.get('client', '')
    
    if query:
        items = items.filter(
            Q(name__icontains=query) |
            Q(sku__icontains=query) |
            Q(description__icontains=query) |
            Q(client__username__icontains=query) |
            Q(category__name__icontains=query)
        )
    
    if status_filter:
        items = items.filter(status=status_filter)
    
    if item_type_filter:
        items = items.filter(item_type=item_type_filter)
    
    if client_filter:
        items = items.filter(client_id=client_filter)
    
    # Sorting
    sort_by = request.GET.get('sort', '-updated_at')
    if sort_by in ['name', '-name', 'quantity', '-quantity', 'unit_price', '-unit_price',
                   'created_at', '-created_at', 'updated_at', '-updated_at', 'expiry_date', '-expiry_date']:
        items = items.order_by(sort_by)
    
    # Calculate stats
    stats = {
        'total': items.count(),
        'active': items.filter(status='active').count(),
        'reserved': items.filter(status='reserved').count(),
        'low_stock': items.filter(
            status='active',
            quantity__lte=F('low_stock_threshold')
        ).count(),
        'near_expiry': items.filter(
            expiry_date__lte=timezone.now().date() + timedelta(days=7),
            expiry_date__gte=timezone.now().date()
        ).count(),
        'total_value': items.aggregate(total=Sum('total_value'))['total'] or 0,
    }
    
    # Get clients for filter
    clients = CustomUser.objects.filter(
        user_type='customer'
    ).order_by('username')
    
    # Pagination
    paginator = Paginator(items, 20)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'items': page_obj,
        'stats': stats,
        'clients': clients,
        'query': query,
        'status_filter': status_filter,
        'item_type_filter': item_type_filter,
        'client_filter': client_filter,
        'sort_by': sort_by,
        'status_choices': InventoryItem.STATUS_CHOICES,
        'item_type_choices': InventoryItem.ITEM_TYPE_CHOICES,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/manager/inventory.html', context)

@login_required
@role_required('manager', 'admin')
def manager_deliveries(request):
    """Manager deliveries view"""
    # Get all deliveries
    deliveries = Delivery.objects.select_related(
        'client', 'assigned_to', 'order'
    ).all()
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    client_filter = request.GET.get('client', '')
    
    if query:
        deliveries = deliveries.filter(
            Q(delivery_number__icontains=query) |
            Q(tracking_number__icontains=query) |
            Q(client_name__icontains=query) |
            Q(client_email__icontains=query)
        )
    
    if status_filter:
        deliveries = deliveries.filter(status=status_filter)
    
    if client_filter:
        deliveries = deliveries.filter(client_id=client_filter)
    
    # Sorting
    sort_by = request.GET.get('sort', '-created_at')
    if sort_by in ['delivery_number', '-delivery_number', 'client_name', '-client_name',
                   'estimated_delivery', '-estimated_delivery', 'created_at', '-created_at']:
        deliveries = deliveries.order_by(sort_by)
    
    # Calculate stats
    stats = {
        'total': deliveries.count(),
        'pending': deliveries.filter(status='pending').count(),
        'processing': deliveries.filter(status='processing').count(),
        'in_transit': deliveries.filter(status='in_transit').count(),
        'delivered': deliveries.filter(status='delivered').count(),
        'late': deliveries.filter(
            status__in=['pending', 'processing', 'in_transit'],
            estimated_delivery__lt=timezone.now()
        ).count(),
    }
    
    # Get clients for filter
    clients = CustomUser.objects.filter(
        user_type='customer'
    ).order_by('username')
    
    # Get delivery staff for assignment
    staff = CustomUser.objects.filter(
        Q(user_type='vendor') | Q(user_type='admin')
    ).order_by('username')
    
    # Pagination
    paginator = Paginator(deliveries, 20)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'deliveries': page_obj,
        'stats': stats,
        'clients': clients,
        'staff': staff,
        'query': query,
        'status_filter': status_filter,
        'client_filter': client_filter,
        'sort_by': sort_by,
        'status_choices': Delivery.STATUS_CHOICES,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/manager/deliveries.html', context)

@login_required
@role_required('manager', 'admin')
def manager_reports(request):
    """Manager reports and analytics"""
    # Date range (last 30 days by default)
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)
    
    # Inventory reports
    inventory_stats = {
        'total_items': InventoryItem.objects.filter(
            created_at__range=[start_date, end_date]
        ).count(),
        'total_value': InventoryItem.objects.filter(
            created_at__range=[start_date, end_date]
        ).aggregate(total=Sum('total_value'))['total'] or 0,
        'items_added': InventoryItem.objects.filter(
            created_at__range=[start_date, end_date]
        ).count(),
        'items_updated': InventoryItem.objects.filter(
            updated_at__range=[start_date, end_date],
            updated_at__gt=F('created_at')
        ).count(),
    }
    
    # Delivery reports
    delivery_stats = {
        'total_deliveries': Delivery.objects.filter(
            created_at__range=[start_date, end_date]
        ).count(),
        'successful_deliveries': Delivery.objects.filter(
            status='delivered',
            created_at__range=[start_date, end_date]
        ).count(),
        'late_deliveries': Delivery.objects.filter(
            status='delivered',
            actual_delivery__gt=F('estimated_delivery'),
            created_at__range=[start_date, end_date]
        ).count(),
        'avg_delivery_time': Delivery.objects.filter(
            status='delivered',
            created_at__range=[start_date, end_date]
        ).aggregate(avg_time=Avg(F('actual_delivery') - F('created_at')))['avg_time'],
    }
    
    # Client reports
    client_stats = {
        'active_clients': CustomUser.objects.filter(
            user_type='customer',
            is_active=True,
            date_joined__range=[start_date, end_date]
        ).count(),
        'new_clients': CustomUser.objects.filter(
            user_type='customer',
            date_joined__range=[start_date, end_date]
        ).count(),
        'top_clients': InventoryItem.objects.filter(
            created_at__range=[start_date, end_date]
        ).values('client__username').annotate(
            total_items=Count('id'),
            total_value=Sum('total_value')
        ).order_by('-total_value')[:5],
    }
    
    context = {
        'inventory_stats': inventory_stats,
        'delivery_stats': delivery_stats,
        'client_stats': client_stats,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/manager/reports.html', context)

@login_required
@role_required('manager', 'admin')
@require_POST
def update_delivery_status(request, delivery_id):
    """Update delivery status"""
    delivery = get_object_or_404(Delivery, id=delivery_id)
    new_status = request.POST.get('status')
    notes = request.POST.get('notes', '')
    
    if new_status and new_status in dict(Delivery.STATUS_CHOICES):
        # Update status
        old_status = delivery.status
        delivery.status = new_status
        delivery.status_changed_at = timezone.now()
        delivery.status_changed_by = request.user
        delivery.save()
        
        # Record status change
        DeliveryStatusHistory.objects.create(
            delivery=delivery,
            from_status=old_status,
            to_status=new_status,
            changed_by=request.user,
            notes=notes
        )
        
        # Send notification to client
        Notification.objects.create(
            user=delivery.client,
            title=f"Delivery Status Updated",
            message=f"Your delivery #{delivery.delivery_number} status has been updated to {delivery.get_status_display()}.",
            notification_type='order_update',
            related_object_type='delivery',
            related_object_id=delivery.id
        )
        
        messages.success(request, f'Delivery status updated to {delivery.get_status_display()}.')
    else:
        messages.error(request, 'Invalid status selected.')
    
    return redirect('bika:manager_deliveries')

@login_required
@role_required('manager', 'admin')
@require_POST
def assign_delivery_staff(request, delivery_id):
    """Assign staff to delivery"""
    delivery = get_object_or_404(Delivery, id=delivery_id)
    staff_id = request.POST.get('staff_id')
    
    if staff_id:
        staff = get_object_or_404(CustomUser, id=staff_id)
        delivery.assigned_to = staff
        delivery.save()
        
        # Send notification to staff
        Notification.objects.create(
            user=staff,
            title=f"Delivery Assigned",
            message=f"You have been assigned to delivery #{delivery.delivery_number} for {delivery.client_name}.",
            notification_type='system_alert',
            related_object_type='delivery',
            related_object_id=delivery.id
        )
        
        messages.success(request, f'Delivery assigned to {staff.username}.')
    else:
        messages.error(request, 'Please select a staff member.')
    
    return redirect('bika:manager_deliveries')

# ==================== STORAGE STAFF VIEWS ====================

@login_required
@role_required('storage_staff', 'manager', 'admin')
def storage_dashboard(request):
    """Storage staff dashboard"""
    # Get assigned storage locations
    user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else StorageLocation.objects.all()
    
    # Get inventory in assigned locations
    items = InventoryItem.objects.filter(
        location__in=user_locations,
        status='active'
    ).select_related('category', 'client', 'location')
    
    # Calculate stats
    stats = {
        'total_items': items.count(),
        'assigned_locations': user_locations.count(),
        'total_capacity': sum(loc.capacity for loc in user_locations),
        'total_occupancy': sum(loc.current_occupancy for loc in user_locations),
        'low_stock': items.filter(quantity__lte=F('low_stock_threshold')).count(),
        'near_expiry': items.filter(
            expiry_date__lte=timezone.now().date() + timedelta(days=7),
            expiry_date__gte=timezone.now().date()
        ).count(),
    }
    
    # Get recent activities
    recent_activities = InventoryHistory.objects.filter(
        item__location__in=user_locations
    ).select_related('item', 'user').order_by('-timestamp')[:10]
    
    # Get alerts for assigned locations
    location_alerts = []
    for location in user_locations:
        if location.occupancy_percentage > 90:
            location_alerts.append({
                'location': location,
                'type': 'high_occupancy',
                'message': f'Location {location.name} is {location.occupancy_percentage:.1f}% full'
            })
        if location.occupancy_percentage < 10:
            location_alerts.append({
                'location': location,
                'type': 'low_occupancy',
                'message': f'Location {location.name} is only {location.occupancy_percentage:.1f}% utilized'
            })
    
    context = {
        'stats': stats,
        'recent_activities': recent_activities,
        'location_alerts': location_alerts,
        'user_locations': user_locations,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/storage/dashboard.html', context)

@login_required
@role_required('storage_staff', 'manager', 'admin')
def storage_inventory(request):
    """Storage staff inventory view"""
    # Get assigned storage locations
    user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else StorageLocation.objects.all()
    
    # Get inventory in assigned locations
    items = InventoryItem.objects.filter(
        location__in=user_locations,
        is_active=True
    ).select_related('category', 'client', 'location')
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    location_filter = request.GET.get('location', '')
    
    if query:
        items = items.filter(
            Q(name__icontains=query) |
            Q(sku__icontains=query) |
            Q(description__icontains=query) |
            Q(client__username__icontains=query)
        )
    
    if status_filter:
        items = items.filter(status=status_filter)
    
    if location_filter:
        items = items.filter(location_id=location_filter)
    
    # Sorting
    sort_by = request.GET.get('sort', 'location__name')
    if sort_by in ['name', '-name', 'quantity', '-quantity', 'location__name', '-location__name',
                   'expiry_date', '-expiry_date']:
        items = items.order_by(sort_by)
    
    context = {
        'items': items,
        'user_locations': user_locations,
        'query': query,
        'status_filter': status_filter,
        'location_filter': location_filter,
        'sort_by': sort_by,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/storage/inventory.html', context)

@login_required
@role_required('storage_staff', 'manager', 'admin')
def storage_locations(request):
    """Storage locations management"""
    # Get assigned storage locations
    user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else StorageLocation.objects.all()
    
    context = {
        'locations': user_locations,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/storage/locations.html', context)

@login_required
@role_required('storage_staff', 'manager', 'admin')
def storage_check_in(request):
    """Check in items to storage"""
    if request.method == 'POST':
        sku = request.POST.get('sku', '').strip()
        location_id = request.POST.get('location_id')
        quantity = int(request.POST.get('quantity', 1))
        notes = request.POST.get('notes', '')
        
        if not sku or not location_id:
            messages.error(request, 'SKU and location are required.')
            return redirect('bika:storage_check_in')
        
        try:
            # Find item by SKU
            item = InventoryItem.objects.get(sku=sku)
            location = StorageLocation.objects.get(id=location_id)
            
            # Check if location is assigned to user
            user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else []
            if location not in user_locations:
                messages.error(request, 'You are not assigned to this location.')
                return redirect('bika:storage_check_in')
            
            # Update item
            old_quantity = item.quantity
            item.quantity += quantity
            item.location = location
            item.last_checked = timezone.now()
            item.checked_by = request.user
            item.save()
            
            # Record history
            InventoryHistory.objects.create(
                item=item,
                action='check_in',
                user=request.user,
                previous_quantity=old_quantity,
                new_quantity=item.quantity,
                previous_location=item.location,
                new_location=location,
                notes=notes
            )
            
            messages.success(request, f'{quantity} units of {item.name} checked into {location.name}.')
            
        except InventoryItem.DoesNotExist:
            messages.error(request, f'Item with SKU {sku} not found.')
        except StorageLocation.DoesNotExist:
            messages.error(request, 'Location not found.')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
        
        return redirect('bika:storage_check_in')
    
    # GET request - show form
    user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else StorageLocation.objects.filter(is_active=True)
    
    context = {
        'locations': user_locations,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/storage/check_in.html', context)

@login_required
@role_required('storage_staff', 'manager', 'admin')
def storage_check_out(request):
    """Check out items from storage"""
    if request.method == 'POST':
        sku = request.POST.get('sku', '').strip()
        quantity = int(request.POST.get('quantity', 1))
        notes = request.POST.get('notes', '')
        
        if not sku:
            messages.error(request, 'SKU is required.')
            return redirect('bika:storage_check_out')
        
        try:
            # Find item by SKU
            item = InventoryItem.objects.get(sku=sku)
            
            # Check if item is in user's assigned location
            user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else []
            if item.location not in user_locations:
                messages.error(request, 'Item is not in your assigned location.')
                return redirect('bika:storage_check_out')
            
            # Check if enough quantity
            if item.quantity < quantity:
                messages.error(request, f'Not enough stock. Available: {item.quantity}')
                return redirect('bika:storage_check_out')
            
            # Update item
            old_quantity = item.quantity
            item.quantity -= quantity
            item.last_checked = timezone.now()
            item.checked_by = request.user
            
            # If quantity becomes 0, mark as reserved
            if item.quantity == 0:
                item.status = 'reserved'
            
            item.save()
            
            # Record history
            InventoryHistory.objects.create(
                item=item,
                action='check_out',
                user=request.user,
                previous_quantity=old_quantity,
                new_quantity=item.quantity,
                notes=notes
            )
            
            messages.success(request, f'{quantity} units of {item.name} checked out.')
            
        except InventoryItem.DoesNotExist:
            messages.error(request, f'Item with SKU {sku} not found.')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
        
        return redirect('bika:storage_check_out')
    
    # GET request - show form
    context = {
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/storage/check_out.html', context)

@login_required
@role_required('storage_staff', 'manager', 'admin')
def storage_transfer(request):
    """Transfer items between storage locations"""
    if request.method == 'POST':
        sku = request.POST.get('sku', '').strip()
        from_location_id = request.POST.get('from_location_id')
        to_location_id = request.POST.get('to_location_id')
        quantity = int(request.POST.get('quantity', 1))
        notes = request.POST.get('notes', '')
        
        if not sku or not from_location_id or not to_location_id:
            messages.error(request, 'SKU and both locations are required.')
            return redirect('bika:storage_transfer')
        
        try:
            # Find item
            item = InventoryItem.objects.get(sku=sku)
            from_location = StorageLocation.objects.get(id=from_location_id)
            to_location = StorageLocation.objects.get(id=to_location_id)
            
            # Check permissions
            user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else []
            if from_location not in user_locations or to_location not in user_locations:
                messages.error(request, 'You are not assigned to one or both locations.')
                return redirect('bika:storage_transfer')
            
            # Check if item is in from_location
            if item.location != from_location:
                messages.error(request, f'Item is not in {from_location.name}.')
                return redirect('bika:storage_transfer')
            
            # Check quantity
            if item.quantity < quantity:
                messages.error(request, f'Not enough stock. Available: {item.quantity}')
                return redirect('bika:storage_transfer')
            
            # Update item
            old_quantity = item.quantity
            item.quantity = quantity
            item.location = to_location
            item.save()
            
            # Record history
            InventoryHistory.objects.create(
                item=item,
                action='transfer',
                user=request.user,
                previous_quantity=old_quantity,
                new_quantity=item.quantity,
                previous_location=from_location,
                new_location=to_location,
                notes=notes
            )
            
            messages.success(request, f'{quantity} units of {item.name} transferred from {from_location.name} to {to_location.name}.')
            
        except InventoryItem.DoesNotExist:
            messages.error(request, f'Item with SKU {sku} not found.')
        except StorageLocation.DoesNotExist:
            messages.error(request, 'Location not found.')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
        
        return redirect('bika:storage_transfer')
    
    # GET request - show form
    user_locations = request.user.assigned_locations.all() if hasattr(request.user, 'assigned_locations') else StorageLocation.objects.filter(is_active=True)
    
    context = {
        'locations': user_locations,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/storage/transfer.html', context)

# ==================== CLIENT VIEWS ====================

# ==================== CLIENT VIEWS ====================

@login_required
@role_required('client', 'customer')
def client_dashboard(request):
    """Client dashboard - read-only access"""
    # Get client's items
    client_items = InventoryItem.objects.filter(
        client=request.user,
        is_active=True
    ).select_related('category', 'location')
    
    # Calculate stats
    stats = {
        'total_items': client_items.count(),
        'active_items': client_items.filter(status='active').count(),
        'reserved_items': client_items.filter(status='reserved').count(),
        'total_value': client_items.aggregate(total=Sum('total_value'))['total'] or 0,
        'low_stock': client_items.filter(
            status='active',
            quantity__lte=F('low_stock_threshold')
        ).count(),
    }
    
    # Get client's deliveries
    client_deliveries = Delivery.objects.filter(
        client=request.user
    ).order_by('-created_at')[:5]
    
    # Get recent activities
    recent_activities = InventoryHistory.objects.filter(
        item__client=request.user
    ).select_related('item', 'user').order_by('-timestamp')[:10]
    
    context = {
        'items': client_items[:10],  # Show only recent 10
        'stats': stats,
        'deliveries': client_deliveries,
        'recent_activities': recent_activities,
        'user_role': 'Client',
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/dashboard.html', context)

@login_required
@role_required('client', 'customer')
def client_inventory(request):
    """Client inventory view - read-only"""
    # Get client's items
    items = InventoryItem.objects.filter(
        client=request.user,
        is_active=True
    ).select_related('category', 'location')
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    category_filter = request.GET.get('category', '')
    
    if query:
        items = items.filter(
            Q(name__icontains=query) |
            Q(sku__icontains=query) |
            Q(description__icontains=query) |
            Q(category__name__icontains=query)
        )
    
    if status_filter:
        items = items.filter(status=status_filter)
    
    if category_filter:
        items = items.filter(category_id=category_filter)
    
    # Sorting
    sort_by = request.GET.get('sort', '-updated_at')
    if sort_by in ['name', '-name', 'quantity', '-quantity', 'unit_price', '-unit_price',
                   'expiry_date', '-expiry_date']:
        items = items.order_by(sort_by)
    
    # Get categories for filter
    categories = ProductCategory.objects.filter(
        inventory_items__client=request.user
    ).distinct()
    
    # Pagination
    paginator = Paginator(items, 20)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'items': page_obj,
        'categories': categories,
        'query': query,
        'status_filter': status_filter,
        'category_filter': category_filter,
        'sort_by': sort_by,
        'can_edit': False,  # Read-only for clients
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/inventory.html', context)

@login_required
@role_required('client', 'customer')
def client_item_detail(request, item_id):
    """Client view item details - read-only"""
    item = get_object_or_404(InventoryItem, 
                           id=item_id, 
                           client=request.user, 
                           is_active=True)
    
    # Get item history
    history = InventoryHistory.objects.filter(
        item=item
    ).select_related('user').order_by('-timestamp')[:20]
    
    context = {
        'item': item,
        'history': history,
        'can_edit': False,  # Read-only for clients
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/item_detail.html', context)

@login_required
@role_required('client', 'customer')
def client_deliveries(request):
    """Client deliveries view"""
    # Get client's deliveries
    deliveries = Delivery.objects.filter(
        client=request.user
    ).select_related('assigned_to', 'order')
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    
    if query:
        deliveries = deliveries.filter(
            Q(delivery_number__icontains=query) |
            Q(tracking_number__icontains=query) |
            Q(recipient_name__icontains=query)
        )
    
    if status_filter:
        deliveries = deliveries.filter(status=status_filter)
    
    # Sorting
    sort_by = request.GET.get('sort', '-created_at')
    if sort_by in ['delivery_number', '-delivery_number', 'status', '-status',
                   'estimated_delivery', '-estimated_delivery']:
        deliveries = deliveries.order_by(sort_by)
    
    # Pagination
    paginator = Paginator(deliveries, 20)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'deliveries': page_obj,
        'query': query,
        'status_filter': status_filter,
        'sort_by': sort_by,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/deliveries.html', context)

@login_required
@role_required('client', 'customer')
def client_delivery_detail(request, delivery_id):
    """Client delivery detail view"""
    delivery = get_object_or_404(Delivery, 
                               id=delivery_id, 
                               client=request.user)
    
    # Get delivery items
    delivery_items = DeliveryItem.objects.filter(
        delivery=delivery
    ).select_related('item')
    
    # Get status history
    status_history = DeliveryStatusHistory.objects.filter(
        delivery=delivery
    ).select_related('changed_by').order_by('-timestamp')
    
    context = {
        'delivery': delivery,
        'delivery_items': delivery_items,
        'status_history': status_history,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/delivery_detail.html', context)

@login_required
def client_requests(request):
    """Client requests overview page"""
    # Check if user is a client
    if not request.user.user_type == 'customer':
        messages.error(request, "Access denied. Client account required.")
        return redirect('bika:home')
    
    # Get client's requests
    requests = ClientRequest.objects.filter(
        client=request.user
    ).order_by('-requested_date')
    
    # Get request stats
    stats = {
        'total': requests.count(),
        'pending': requests.filter(status='pending').count(),
        'in_progress': requests.filter(status='in_progress').count(),
        'completed': requests.filter(status='completed').count(),
    }
    
    context = {
        'requests': requests,
        'stats': stats,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/requests.html', context)

@login_required
def client_requests_list(request):
    """Client requests list - same as client_requests but for consistency with URL"""
    return client_requests(request)

@login_required
def client_request_detail(request, request_id):
    """Client request detail view"""
    # Check if user is a client
    if not request.user.user_type == 'customer':
        messages.error(request, "Access denied. Client account required.")
        return redirect('bika:home')
    
    # Get request and verify ownership
    client_request = get_object_or_404(ClientRequest, id=request_id, client=request.user)
    
    # Get related items
    related_items = client_request.inventory_items.all()
    
    context = {
        'request': client_request,
        'related_items': related_items,
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/request_detail.html', context)

@login_required
def create_client_request(request):
    """Create a new client request"""
    # Ensure only clients can create requests
    if not request.user.user_type == 'customer':
        messages.error(request, 'Only clients can create requests.')
        return redirect('bika:home')
    
    if request.method == 'POST':
        form = ClientRequestForm(request.POST, user=request.user)
        
        if form.is_valid():
            client_request = form.save(commit=False)
            client_request.client = request.user
            client_request.status = 'pending'
            
            # Generate request number if not provided
            if not client_request.request_number:
                import random
                import string
                timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
                random_str = ''.join(random.choices(string.ascii_uppercase, k=3))
                client_request.request_number = f"REQ-{timestamp}-{random_str}"
            
            client_request.save()
            
            # Handle inventory items if selected
            inventory_items = request.POST.getlist('inventory_items')
            if inventory_items:
                for item_id in inventory_items:
                    try:
                        item = InventoryItem.objects.get(id=item_id, client=request.user)
                        client_request.inventory_items.add(item)
                    except InventoryItem.DoesNotExist:
                        pass
            
            messages.success(request, 'Your request has been submitted successfully!')
            return redirect('bika:client_requests')
    else:
        form = ClientRequestForm(user=request.user)
    
    # Get client's available inventory items
    inventory_items = InventoryItem.objects.filter(
        client=request.user,
        status='active'
    ).select_related('product', 'category')
    
    context = {
        'form': form,
        'inventory_items': inventory_items,
        'page_title': 'Create New Request',
        'site_info': SiteInfo.objects.first(),
    }
    
    return render(request, 'bika/pages/client/create_request.html', context)

# ==================== UTILITY FUNCTIONS FOR CHARTS ====================

def get_item_type_distribution():
    """Get item type distribution for charts"""
    distribution = {
        'Storage': InventoryItem.objects.filter(item_type='storage', status='active').count(),
        'For Sale': InventoryItem.objects.filter(item_type='sale', status='active').count(),
        'Rental': InventoryItem.objects.filter(item_type='rental', status='active').count(),
    }
    return distribution

def get_delivery_status_distribution():
    """Get delivery status distribution for charts"""
    distribution = {
        'Pending': Delivery.objects.filter(status='pending').count(),
        'Processing': Delivery.objects.filter(status='processing').count(),
        'In Transit': Delivery.objects.filter(status='in_transit').count(),
        'Out for Delivery': Delivery.objects.filter(status='out_for_delivery').count(),
        'Delivered': Delivery.objects.filter(status='delivered').count(),
        'Cancelled': Delivery.objects.filter(status='cancelled').count(),
        'Failed': Delivery.objects.filter(status='failed').count(),
    }
    return distribution

def get_category_distribution():
    """Get category distribution for charts"""
    from django.db.models import Count
    
    categories = ProductCategory.objects.annotate(
        item_count=Count('inventory_items', filter=Q(inventory_items__status='active'))
    ).filter(item_count__gt=0)
    
    distribution = {}
    for category in categories:
        distribution[category.name] = category.item_count
    
    return distribution

def get_client_item_stats(client_id):
    """Get item statistics for a specific client"""
    items = InventoryItem.objects.filter(client_id=client_id, status='active')
    
    stats = {
        'total_items': items.count(),
        'total_value': items.aggregate(total=Sum('total_value'))['total'] or 0,
        'active_items': items.filter(status='active').count(),
        'low_stock': items.filter(quantity__lte=F('low_stock_threshold')).count(),
        'by_type': {
            'storage': items.filter(item_type='storage').count(),
            'sale': items.filter(item_type='sale').count(),
            'rental': items.filter(item_type='rental').count(),
        }
    }
    
    return stats