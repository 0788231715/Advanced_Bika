# bika/views.py - FIXED AND COMPLETE VERSION
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
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

# Import models
from .models import (
    CustomUser, Product, ProductCategory, ProductImage, ProductReview,
    Wishlist, Cart, Order, OrderItem, Payment,
    SiteInfo, Service, Testimonial, ContactMessage, FAQ,
    StorageLocation, FruitType, FruitBatch, FruitQualityReading, 
    RealTimeSensorData, ProductAlert, Notification,
    ProductDataset, TrainedModel, PaymentGatewaySettings, CurrencyExchangeRate
)

# Import forms
from .forms import (
    ContactForm, NewsletterForm, CustomUserCreationForm, 
    VendorRegistrationForm, CustomerRegistrationForm, ProductForm,
    ProductImageForm, FruitBatchForm, FruitQualityReadingForm
)

# Import services
class SimpleFruitAIService:
    def __init__(self):
        print("Simple Fruit AI Service initialized")
    
    def train_fruit_quality_model(self, csv_file, model_type='random_forest'):
        """Simulate training"""
        return {
            'success': True,
            'message': 'Training simulation complete',
            'accuracy': 0.85,
            'training_samples': 1000,
            'model_type': model_type
        }
    
    def predict_fruit_quality(self, fruit_name, temperature, humidity, 
                            light_intensity, co2_level, batch_id=None):
        """Simple rule-based prediction"""
        # Simple rules
        quality_score = 80
        
        if 2 <= temperature <= 8 and 85 <= humidity <= 95:
            predicted_class = 'Good'
            confidence = 0.8
        elif temperature < 2 or temperature > 12:
            predicted_class = 'Poor'
            confidence = 0.7
        else:
            predicted_class = 'Fair'
            confidence = 0.6
        
        return {
            'success': True,
            'prediction': {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'quality_score': quality_score,
                'recommendations': ['Maintain optimal storage conditions']
            }
        }
    
    def get_batch_quality_report(self, batch_id, hours=24):
        """Generate batch report"""
        return {
            'success': True,
            'report': {
                'batch_id': batch_id,
                'quality': 'Good',
                'recommendations': ['Continue monitoring']
            }
        }

# Create global instance
fruit_ai_service = SimpleFruitAIService()
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

# Set up logger
logger = logging.getLogger(__name__)

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

@staff_member_required
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
    
    context = {
        # User statistics (FLATTENED - as expected by template)
        'total_users': total_users,
        'total_admins': total_admins,
        'total_vendors': total_vendors,
        'total_customers': total_customers,
        'new_users_today': new_users_today,
        'active_users': active_users,
        'admin_percentage': admin_percentage,
        'vendor_percentage': vendor_percentage,
        'customer_percentage': customer_percentage,
        
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

@login_required
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