from asyncio.log import logger
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Count, Q, F
from django.utils import timezone
from datetime import timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.views.generic import ListView, DetailView, TemplateView
from django.contrib.auth import login, authenticate, logout  # ADDED logout here
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
import json
import pandas as pd
import datetime

# Import models
from .models import (
    Notification, Payment, ProductAlert, ProductDataset, RealTimeSensorData, SiteInfo, Service, StorageLocation, Testimonial, ContactMessage, FAQ,
    CustomUser, Product, ProductCategory, ProductImage, ProductReview,
    Wishlist, Cart, Order, OrderItem
)

# Import forms
from .forms import (
    ContactForm, NewsletterForm, CustomUserCreationForm, 
    VendorRegistrationForm, CustomerRegistrationForm, ProductForm,
    ProductImageForm
)
from bika import models

from .models import (
    ProductDataset, SiteInfo, Service, Testimonial, ContactMessage, FAQ,
    CustomUser, Product, ProductCategory, ProductImage, ProductReview,
    Wishlist, Cart, Order, OrderItem, StorageLocation, ProductAlert, 
    Notification, RealTimeSensorData, TrainedModel  # ADD THESE
)
# ... other imports ...

try:
    from bika.notification import RealNotificationService
    from bika.service import RealProductAIService
    AI_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"AI services not available: {e}")
    AI_SERVICES_AVAILABLE = False
    # Create dummy classes for fallback
    class RealNotificationService:
        def __init__(self):
            pass
        def run_daily_analysis(self):
            print("AI services not available - running in fallback mode")
    
    class RealProductAIService:
        def __init__(self):
            pass

class HomeView(TemplateView):
    template_name = 'bika/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Existing services and testimonials
        context['featured_services'] = Service.objects.filter(is_active=True)[:6]
        context['featured_testimonials'] = Testimonial.objects.filter(
            is_active=True, 
            is_featured=True
        )[:3]
        context['faqs'] = FAQ.objects.filter(is_active=True)[:5]
        
        # Add featured products with error handling
        try:
            featured_products = Product.objects.filter(
                status='active',
                is_featured=True
            ).select_related('category', 'vendor')[:8]
            
            # Add primary images to products
            for product in featured_products:
                try:
                    product.primary_image = product.images.filter(is_primary=True).first()
                    if not product.primary_image:
                        product.primary_image = product.images.first()
                except Exception:
                    product.primary_image = None
            
            context['featured_products'] = featured_products
            
        except Exception as e:
            print(f"Error loading featured products: {e}")
            context['featured_products'] = []
        
        # Add site info if available
        try:
            context['site_info'] = SiteInfo.objects.first()
        except Exception:
            context['site_info'] = None
        
        return context

def about_view(request):
    services = Service.objects.filter(is_active=True)
    testimonials = Testimonial.objects.filter(is_active=True)[:4]
    
    context = {
        'services': services,
        'testimonials': testimonials,
    }
    return render(request, 'bika/pages/about.html', context)

def services_view(request):
    services = Service.objects.filter(is_active=True)
    return render(request, 'bika/pages/services.html', {'services': services})

class ServiceDetailView(DetailView):
    model = Service
    template_name = 'bika/pages/service_detail.html'
    context_object_name = 'service'
    slug_field = 'slug'
    slug_url_kwarg = 'slug'

def contact_view(request):
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
            
            # Send email notification (optional)
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
                print(f"Email error: {e}")
            
            messages.success(
                request, 
                'Thank you for your message! We will get back to you soon.'
            )
            return redirect('bika:contact')
    else:
        form = ContactForm()
    
    return render(request, 'bika/pages/contact.html', {'form': form})

def faq_view(request):
    faqs = FAQ.objects.filter(is_active=True)
    return render(request, 'bika/pages/faq.html', {'faqs': faqs})

def newsletter_subscribe(request):
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        form = NewsletterForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            return JsonResponse({
                'success': True,
                'message': 'Thank you for subscribing to our newsletter!'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Please enter a valid email address.'
            })
    return JsonResponse({'success': False, 'message': 'Invalid request'})

@staff_member_required
def admin_dashboard(request):
    """Enhanced admin dashboard with user and product statistics"""
    # Get current date and time
    now = timezone.now()
    today = now.date()
    today_start = timezone.make_aware(timezone.datetime.combine(today, timezone.datetime.min.time()))
    thirty_days_ago = now - timedelta(days=30)

    # User Statistics
    total_users = CustomUser.objects.count()
    total_admins = CustomUser.objects.filter(user_type='admin').count()
    total_vendors = CustomUser.objects.filter(user_type='vendor').count()
    total_customers = CustomUser.objects.filter(user_type='customer').count()
    
    # Calculate percentages
    admin_percentage = round((total_admins / total_users * 100), 2) if total_users > 0 else 0
    vendor_percentage = round((total_vendors / total_users * 100), 2) if total_users > 0 else 0
    customer_percentage = round((total_customers / total_users * 100), 2) if total_users > 0 else 0
    
    # Active users (logged in last 30 days)
    active_users = CustomUser.objects.filter(last_login__gte=thirty_days_ago).count()
    active_users_percentage = round((active_users / total_users * 100), 2) if total_users > 0 else 0
    
    # New users today
    new_users_today = CustomUser.objects.filter(date_joined__gte=today_start).count()

    # Product Statistics
    total_products = Product.objects.count()
    active_products = Product.objects.filter(status='active').count()
    draft_products = Product.objects.filter(status='draft').count()
    
    # Calculate active products percentage
    active_products_percentage = round((active_products / total_products * 100), 2) if total_products > 0 else 0
    
    # Inventory alerts - FIXED: Use F() directly (not models.F)
    low_stock_products = Product.objects.filter(
        stock_quantity__lte=F('low_stock_threshold'),  # CHANGED: models.F to F
        track_inventory=True,
        stock_quantity__gt=0
    ).count()
    
    out_of_stock_products = Product.objects.filter(
        stock_quantity=0,
        track_inventory=True
    ).count()
    
    low_stock_count = low_stock_products + out_of_stock_products

    # Vendor Statistics
    active_vendors = CustomUser.objects.filter(
        user_type='vendor', 
        is_active=True,
        product__status='active'
    ).distinct().count()
    
    active_vendors_percentage = round((active_vendors / total_vendors * 100), 2) if total_vendors > 0 else 0

    # Order Statistics
    total_orders = Order.objects.count()
    pending_orders = Order.objects.filter(status='pending').count()
    
    # Calculate revenue manually without SUM
    total_revenue = 0
    today_revenue = 0
    
    # Calculate total revenue manually
    all_orders = Order.objects.all()
    for order in all_orders:
        if order.total_amount:
            try:
                total_revenue += float(order.total_amount)
            except (TypeError, ValueError):
                continue
    
    # Calculate today's revenue
    today_orders = Order.objects.filter(created_at__gte=today_start)
    for order in today_orders:
        if order.total_amount:
            try:
                today_revenue += float(order.total_amount)
            except (TypeError, ValueError):
                continue

    # Format revenue for display
    total_revenue_display = "{:,.2f}".format(total_revenue)
    today_revenue_display = "{:,.2f}".format(today_revenue)

    # Category Statistics
    total_categories = ProductCategory.objects.count()
    active_categories = ProductCategory.objects.filter(is_active=True).count()

    # Recent data
    recent_products = Product.objects.select_related('vendor', 'category').prefetch_related('images').order_by('-created_at')[:6]
    recent_messages = ContactMessage.objects.filter(status='new').order_by('-submitted_at')[:5]

    # Existing stats for compatibility
    total_services = Service.objects.count()
    total_testimonials = Testimonial.objects.count()
    total_messages = ContactMessage.objects.count()
    new_messages = ContactMessage.objects.filter(status='new').count()
    active_services_count = Service.objects.filter(is_active=True).count()
    featured_testimonials_count = Testimonial.objects.filter(is_featured=True, is_active=True).count()
    active_faqs_count = FAQ.objects.filter(is_active=True).count()

    # Get Django and Python version dynamically
    import django
    import sys
    django_version = django.get_version()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    context = {
        # Enhanced User Stats
        'total_users': total_users,
        'total_admins': total_admins,
        'total_vendors': total_vendors,
        'total_customers': total_customers,
        'admin_percentage': admin_percentage,
        'vendor_percentage': vendor_percentage,
        'customer_percentage': customer_percentage,
        'active_users': active_users,
        'active_users_percentage': active_users_percentage,
        'new_users_today': new_users_today,
        
        # Enhanced Product Stats
        'total_products': total_products,
        'active_products': active_products,
        'active_products_percentage': active_products_percentage,
        'draft_products': draft_products,
        'low_stock_products': low_stock_products,
        'out_of_stock_products': out_of_stock_products,
        'low_stock_count': low_stock_count,
        
        # Vendor Stats
        'active_vendors': active_vendors,
        'active_vendors_percentage': active_vendors_percentage,
        
        # Order Stats
        'total_orders': total_orders,
        'pending_orders': pending_orders,
        'total_revenue': total_revenue_display,
        'today_revenue': today_revenue_display,
        
        # Category Stats
        'total_categories': total_categories,
        'active_categories': active_categories,
        
        # Recent Data
        'recent_products': recent_products,
        'recent_messages': recent_messages,
        
        # Existing stats for compatibility
        'total_services': total_services,
        'total_testimonials': total_testimonials,
        'total_messages': total_messages,
        'new_messages': new_messages,
        'active_services_count': active_services_count,
        'featured_testimonials_count': featured_testimonials_count,
        'active_faqs_count': active_faqs_count,
        
        # System info
        'django_version': django_version,
        'python_version': python_version,
        'debug': settings.DEBUG,
    }
    
    return render(request, 'bika/pages/admin/dashboard.html', context)

def product_list_view(request):
    """Display all active products with filtering and pagination"""
    products = Product.objects.filter(status='active').select_related('category', 'vendor')
    
    # Get filter parameters
    category_slug = request.GET.get('category')
    query = request.GET.get('q', '')
    sort_by = request.GET.get('sort', 'newest')
    
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
            Q(category__name__icontains=query)
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
    
    # Get categories for sidebar
    categories = ProductCategory.objects.filter(is_active=True).annotate(
        product_count=Count('products', filter=Q(products__status='active'))
    )
    
    # Count active vendors
    active_vendors = CustomUser.objects.filter(
        user_type='vendor', 
        product__status='active'
    ).distinct().count()
    
    context = {
        'products': page_obj,
        'categories': categories,
        'current_category': current_category,
        'query': query,
        'total_products': products.count(),
        'active_vendors': active_vendors,
    }
    return render(request, 'bika/pages/products.html', context)

def product_detail_view(request, slug):
    """Display single product details with enhanced functionality"""
    product = get_object_or_404(Product, slug=slug, status='active')
    
    # Safely increment view count
    product.views_count += 1
    product.save()
    
    # Get related products
    related_products = Product.objects.filter(
        category=product.category,
        status='active'
    ).exclude(id=product.id)[:4]
    
    # Get product reviews
    reviews = ProductReview.objects.filter(product=product, is_approved=True).order_by('-created_at')
    
    # Check if product is in user's wishlist
    in_wishlist = False
    if request.user.is_authenticated:
        in_wishlist = Wishlist.objects.filter(user=request.user, product=product).exists()
    
    # Check if product is in user's cart
    in_cart = False
    cart_quantity = 0
    if request.user.is_authenticated:
        cart_item = Cart.objects.filter(user=request.user, product=product).first()
        if cart_item:
            in_cart = True
            cart_quantity = cart_item.quantity
    
    context = {
        'product': product,
        'related_products': related_products,
        'reviews': reviews,
        'in_wishlist': in_wishlist,
        'in_cart': in_cart,
        'cart_quantity': cart_quantity,
    }
    return render(request, 'bika/pages/product_detail.html', context)

def products_by_category_view(request, category_slug):
    """Display products by category"""
    category = get_object_or_404(ProductCategory, slug=category_slug, is_active=True)
    products = Product.objects.filter(category=category, status='active')
    
    # Get categories for sidebar
    categories = ProductCategory.objects.filter(is_active=True).annotate(
        product_count=Count('products', filter=Q(products__status='active'))
    )
    
    context = {
        'category': category,
        'products': products,
        'categories': categories,
        'current_category': category,
        'total_products': products.count(),
    }
    return render(request, 'bika/pages/products.html', context)

def product_search_view(request):
    """Handle product search"""
    query = request.GET.get('q', '')
    products = Product.objects.filter(status='active')
    
    if query:
        products = products.filter(
            Q(name__icontains=query) | 
            Q(description__icontains=query) |
            Q(short_description__icontains=query) |
            Q(tags__icontains=query)
        )
    
    context = {
        'products': products,
        'query': query,
        'categories': ProductCategory.objects.filter(is_active=True),
    }
    return render(request, 'bika/pages/product_search.html', context)

@login_required
def vendor_dashboard(request):
    """Vendor dashboard"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor account required.")
        return redirect('bika:home')
    
    # Get vendor's products (for staff, show all products)
    if request.user.is_staff:
        vendor_products = Product.objects.all()
    else:
        vendor_products = Product.objects.filter(vendor=request.user)
    
    # Recent orders (you'll need to implement this based on your Order model)
    recent_orders = Order.objects.none()  # Placeholder
    
    context = {
        'total_products': vendor_products.count(),
        'active_products': vendor_products.filter(status='active').count(),
        'draft_products': vendor_products.filter(status='draft').count(),
        'recent_products': vendor_products.order_by('-created_at')[:5],
        'recent_orders': recent_orders,
    }
    return render(request, 'bika/pages/vendor/dashboard.html', context)

@login_required
def vendor_product_list(request):
    """Vendor's product list"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor account required.")
        return redirect('bika:home')
    
    # For staff, show all products; for vendors, show only their products
    if request.user.is_staff:
        products = Product.objects.all()
    else:
        products = Product.objects.filter(vendor=request.user)
    
    context = {
        'products': products,
    }
    return render(request, 'bika/pages/vendor/products.html', context)

@login_required
def vendor_add_product(request):
    """Vendor add product form with multiple image upload"""
    # Allow both vendors and staff to add products
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor or admin account required.")
        return redirect('bika:home')
    
    if request.method == 'POST':
        product_form = ProductForm(request.POST, request.FILES)
        
        if product_form.is_valid():
            # Save product with vendor
            product = product_form.save(commit=False)
            product.vendor = request.user
            
            # Set status based on button clicked
            if 'save_draft' in request.POST:
                product.status = 'draft'
                message = f'Product "{product.name}" saved as draft!'
            else:  # publish button
                product.status = 'active'
                message = f'Product "{product.name}" published successfully!'
            
            product.save()
            
            # Handle multiple images
            images = request.FILES.getlist('images')
            for i, image in enumerate(images):
                ProductImage.objects.create(
                    product=product,
                    image=image,
                    alt_text=product.name,
                    display_order=i,
                    is_primary=(i == 0)  # First image is primary
                )
            
            messages.success(request, message)
            return redirect('bika:vendor_product_list')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        # Set initial status to draft for new products
        product_form = ProductForm(initial={'status': 'draft', 'condition': 'new'})
    
    context = {
        'form': product_form,
        'title': 'Add New Product'
    }
    return render(request, 'bika/pages/vendor/add_product.html', context)

@login_required
def vendor_edit_product(request, product_id):
    """Edit existing product"""
    # For staff, allow editing any product; for vendors, only their own products
    if request.user.is_staff:
        product = get_object_or_404(Product, id=product_id)
    else:
        product = get_object_or_404(Product, id=product_id, vendor=request.user)
    
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)
        if form.is_valid():
            form.save()
            messages.success(request, f'Product "{product.name}" updated successfully!')
            return redirect('bika:vendor_product_list')
    else:
        form = ProductForm(instance=product)
    
    context = {
        'form': form,
        'product': product,
        'title': 'Edit Product'
    }
    return render(request, 'bika/pages/vendor/add_product.html', context)

def vendor_register_view(request):
    """Special vendor registration"""
    # Only redirect logged-in users who are ALREADY vendors
    if request.user.is_authenticated and request.user.is_vendor():
        messages.info(request, "You are already a registered vendor!")
        return redirect('bika:vendor_dashboard')
    
    # Show warning for logged-in customers but still show the form
    if request.user.is_authenticated and not request.user.is_vendor():
        messages.warning(request, "You already have a customer account. Please contact support to convert to vendor.")
    
    if request.method == 'POST':
        form = VendorRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Auto-login after registration
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f"Vendor account created successfully! Welcome to Bika, {user.business_name}.")
                return redirect('bika:vendor_dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = VendorRegistrationForm()
    
    return render(request, 'bika/pages/registration/vendor_register.html', {'form': form})

def register_view(request):
    """User registration view"""
    if request.user.is_authenticated:
        messages.info(request, "You are already logged in!")
        return redirect('bika:home')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Auto-login after registration
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Account created successfully! Welcome to Bika, {username}.')
                return redirect('bika:home')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'bika/pages/registration/register.html', {'form': form})

# User profile views (keep your existing implementations)
@login_required
def user_profile(request):
    """User profile page"""
    user = request.user
    recent_orders = Order.objects.filter(user=user).order_by('-created_at')[:5]
    
    context = {
        'user': user,
        'recent_orders': recent_orders,
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
        
        user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('bika:user_profile')
    
    return redirect('bika:user_profile')

@login_required
def user_orders(request):
    """User orders page"""
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'orders': orders,
    }
    return render(request, 'bika/pages/user/orders.html', context)

@login_required
def order_detail(request, order_id):
    """Order detail page"""
    order = get_object_or_404(Order, id=order_id, user=request.user)
    
    context = {
        'order': order,
    }
    return render(request, 'bika/pages/user/order_detail.html', context)

@login_required
def wishlist(request):
    """User wishlist page"""
    wishlist_items = Wishlist.objects.filter(user=request.user).select_related('product')
    
    context = {
        'wishlist_items': wishlist_items,
    }
    return render(request, 'bika/pages/user/wishlist.html', context)

@login_required
def add_to_wishlist(request, product_id):
    """Add product to wishlist"""
    product = get_object_or_404(Product, id=product_id)
    wishlist_item, created = Wishlist.objects.get_or_create(
        user=request.user,
        product=product
    )
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'message': 'Product added to wishlist!',
            'wishlist_count': Wishlist.objects.filter(user=request.user).count()
        })
    
    messages.success(request, 'Product added to wishlist!')
    return redirect('bika:wishlist')

@login_required
def remove_from_wishlist(request, product_id):
    """Remove product from wishlist"""
    product = get_object_or_404(Product, id=product_id)
    Wishlist.objects.filter(user=request.user, product=product).delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'message': 'Product removed from wishlist!',
            'wishlist_count': Wishlist.objects.filter(user=request.user).count()
        })
    
    messages.success(request, 'Product removed from wishlist!')
    return redirect('bika:wishlist')

@login_required
def cart(request):
    """Shopping cart page"""
    cart_items = Cart.objects.filter(user=request.user).select_related('product')
    total_price = sum(item.total_price for item in cart_items)
    
    context = {
        'cart_items': cart_items,
        'total_price': total_price,
    }
    return render(request, 'bika/pages/user/cart.html', context)

@login_required
def add_to_cart(request, product_id):
    """Add product to cart"""
    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))
    
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
            'cart_count': cart_count
        })
    
    messages.success(request, 'Product added to cart!')
    return redirect('bika:cart')

@login_required
def update_cart(request, product_id):
    """Update cart item quantity"""
    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))
    
    if quantity > 0:
        cart_item = get_object_or_404(Cart, user=request.user, product=product)
        cart_item.quantity = quantity
        cart_item.save()
    else:
        Cart.objects.filter(user=request.user, product=product).delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        cart_items = Cart.objects.filter(user=request.user)
        total_price = sum(item.total_price for item in cart_items)
        return JsonResponse({
            'success': True,
            'total_price': total_price,
            'item_total': cart_item.total_price if quantity > 0 else 0
        })
    
    return redirect('bika:cart')

@login_required
def remove_from_cart(request, product_id):
    """Remove product from cart"""
    product = get_object_or_404(Product, id=product_id)
    Cart.objects.filter(user=request.user, product=product).delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        cart_items = Cart.objects.filter(user=request.user)
        total_price = sum(item.total_price for item in cart_items)
        return JsonResponse({
            'success': True,
            'total_price': total_price,
            'cart_count': cart_items.count()
        })
    
    messages.success(request, 'Product removed from cart!')
    return redirect('bika:cart')

@login_required
def user_settings(request):
    """User settings page"""
    if request.method == 'POST':
        # Handle settings update
        user = request.user
        user.email_notifications = request.POST.get('email_notifications') == 'on'
        user.sms_notifications = request.POST.get('sms_notifications') == 'on'
        user.newsletter_subscription = request.POST.get('newsletter_subscription') == 'on'
        user.save()
        
        messages.success(request, 'Settings updated successfully!')
        return redirect('bika:user_settings')
    
    context = {
        'user': request.user,
    }
    return render(request, 'bika/pages/user/settings.html', context)

# Error handlers
def handler404(request, exception):
    return render(request, 'bika/pages/404.html', status=404)

def handler500(request):
    return render(request, 'bika/pages/500.html', status=500)

def custom_404(request, exception):
    return render(request, 'bika/pages/404.html', status=404)

def custom_500(request):
    return render(request, 'bika/pages/500.html', status=500)



@csrf_exempt
@require_http_methods(["POST"])
def upload_dataset(request):
    """Upload real dataset for training"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    try:
        dataset_file = request.FILES['dataset_file']
        dataset_type = request.POST['dataset_type']
        name = request.POST['name']
        description = request.POST.get('description', '')
        
        # Validate file type
        if not dataset_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are supported'}, status=400)
        
        # Read and validate dataset
        df = pd.read_csv(dataset_file)
        
        # Create dataset record
        dataset = ProductDataset.objects.create(
            name=name,
            dataset_type=dataset_type,
            description=description,
            data_file=dataset_file,
            columns=list(df.columns),
            row_count=len(df)
        )
        
        return JsonResponse({
            'success': True,
            'dataset_id': dataset.id,
            'columns': list(df.columns),
            'row_count': len(df)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def train_model(request):
    """Train model on uploaded dataset"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    try:
        data = json.loads(request.body)
        dataset_id = data['dataset_id']
        model_type = data['model_type']
        
        ai_service = RealProductAIService()
        trained_model = ai_service.train_anomaly_detection_model(dataset_id)
        
        if trained_model:
            return JsonResponse({
                'success': True,
                'model_id': trained_model.id,
                'model_name': trained_model.name
            })
        else:
            return JsonResponse({'error': 'Model training failed'}, status=400)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def receive_sensor_data(request):
    """Receive real sensor data from embedded systems"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['product_barcode', 'sensor_type', 'value', 'location_id']
        for field in required_fields:
            if field not in data:
                return JsonResponse({'error': f'Missing field: {field}'}, status=400)
        
        # Get product and location
        product = Product.objects.get(barcode=data['product_barcode'])
        location = StorageLocation.objects.get(id=data['location_id'])
        
        # Save sensor reading
        sensor_reading = RealTimeSensorData.objects.create(
            product=product,
            sensor_type=data['sensor_type'],
            value=data['value'],
            unit=data.get('unit', ''),
            location=location
        )
        
        # Analyze for alerts
        ai_service = RealProductAIService()
        alerts = ai_service.analyze_sensor_data([sensor_reading])
        
        # Process alerts
        if alerts:
            notification_service = RealNotificationService()
            notification_service.process_sensor_alerts(alerts)
        
        return JsonResponse({'status': 'success', 'alerts_generated': len(alerts)})
        
    except Product.DoesNotExist:
        return JsonResponse({'error': 'Product not found'}, status=404)
    except StorageLocation.DoesNotExist:
        return JsonResponse({'error': 'Location not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    
# Add these to your views.py

@login_required
def notifications_view(request):
    """User notifications page"""
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    unread_count = notifications.filter(is_read=False).count()
    
    context = {
        'notifications': notifications,
        'unread_count': unread_count,
    }
    return render(request, 'bika/pages/user/notifications.html', context)

@login_required
def mark_notification_read(request, notification_id):
    """Mark notification as read"""
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.is_read = True
    notification.save()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('bika:notifications')

@login_required
def unread_notifications_count(request):
    """API endpoint for unread notifications count"""
    if request.user.is_authenticated:
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
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

@staff_member_required
def storage_sites(request):
    """Storage sites management"""
    sites = StorageLocation.objects.all()
    
    context = {
        'sites': sites,
    }
    return render(request, 'bika/pages/admin/storage_sites.html', context)

@login_required
def track_my_products(request):
    """Track vendor's products"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect('bika:home')
    
    # Get vendor's products with alerts
    if request.user.is_staff:
        products = Product.objects.all()
        alerts = ProductAlert.objects.filter(is_resolved=False)
    else:
        products = Product.objects.filter(vendor=request.user)
        alerts = ProductAlert.objects.filter(product__vendor=request.user, is_resolved=False)
    
    context = {
        'products': products,
        'alerts': alerts,
    }
    return render(request, 'bika/pages/vendor/track_products.html', context)

@login_required
def scan_product(request):
    """Product scanning interface"""
    return render(request, 'bika/pages/scan_product.html')

# API endpoints for mobile/scanner integration
@csrf_exempt
@require_http_methods(["GET"])
def api_product_detail(request, barcode):
    """API endpoint for product details by barcode"""
    try:
        product = Product.objects.get(barcode=barcode)
        product_data = {
            'id': product.id,
            'name': product.name,
            'barcode': product.barcode,
            'sku': product.sku,
            'price': str(product.price),
            'stock_quantity': product.stock_quantity,
            'status': product.status,
            'vendor': product.vendor.business_name,
            'category': product.category.name,
        }
        return JsonResponse(product_data)
    except Product.DoesNotExist:
        return JsonResponse({'error': 'Product not found'}, status=404)

@login_required
def mark_all_notifications_read(request):
    """Mark all notifications as read for the current user"""
    Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    messages.success(request, 'All notifications marked as read!')
    return redirect('bika:notifications')

@login_required
def delete_notification(request, notification_id):
    """Delete a specific notification"""
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    messages.success(request, 'Notification deleted!')
    return redirect('bika:notifications')

@never_cache
@csrf_protect
def custom_logout(request):
    """Proper logout that clears session and prevents back button access"""
    # Store some info before logout for feedback
    username = request.user.username if request.user.is_authenticated else "User"
    
    # Logout the user - this clears the session
    logout(request)
    
    # Create redirect response
    response = redirect('bika:logout_success')
    
    # Completely clear the session cookie
    request.session.flush()  # Remove session from database
    response.delete_cookie('sessionid')
    response.delete_cookie('csrftoken')
    
    # Set security headers to prevent caching
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = 'Fri, 01 Jan 1990 00:00:00 GMT'
    response['X-Content-Type-Options'] = 'nosniff'
    response['X-Frame-Options'] = 'DENY'
    
    # Add message for feedback
    from django.contrib import messages
    messages.success(request, f'Goodbye {username}! You have been successfully logged out.')
    
    return response

def logout_success(request):
    """Logout success page with proper security headers"""
    response = render(request, 'bika/pages/registration/logout.html')
    
    # Ensure no caching of this page
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = 'Fri, 01 Jan 1990 00:00:00 GMT'
    
    return response
# UPDATE THE CUSTOM LOGOUT FUNCTION - FIX THE DECORATORS
@never_cache
@csrf_protect
@login_required
def custom_logout(request):
    """Proper logout that clears session and prevents back button access"""
    # Store some info before logout for feedback
    username = request.user.username
    
    # Logout the user - this clears the session
    logout(request)
    
    # Create redirect response
    response = redirect('bika:logout_success')
    
    # Completely clear the session cookie
    request.session.flush()  # Remove session from database
    response.delete_cookie('sessionid')
    response.delete_cookie('csrftoken')
    
    # Set security headers to prevent caching
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = 'Fri, 01 Jan 1990 00:00:00 GMT'
    response['X-Content-Type-Options'] = 'nosniff'
    response['X-Frame-Options'] = 'DENY'
    
    # Add message for feedback
    messages.success(request, f'Goodbye {username}! You have been successfully logged out.')
    
    return response

def logout_success(request):
    """Logout success page with proper security headers"""
    response = render(request, 'bika/pages/registration/logout.html')
    
    # Ensure no caching of this page
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = 'Fri, 01 Jan 1990 00:00:00 GMT'
    
    return response

# ADD THIS FUNCTION FOR PRODUCT DETAIL PAGE
def product_detail_view(request, slug):
    """Display single product details with enhanced functionality"""
    product = get_object_or_404(Product, slug=slug, status='active')
    
    # Increment view count
    product.views_count += 1
    product.save()
    
    # Get related products
    related_products = Product.objects.filter(
        category=product.category,
        status='active'
    ).exclude(id=product.id)[:4]
    
    # Get product reviews
    reviews = ProductReview.objects.filter(product=product, is_approved=True).order_by('-created_at')
    
    # Check if product is in user's wishlist
    in_wishlist = False
    if request.user.is_authenticated:
        in_wishlist = Wishlist.objects.filter(user=request.user, product=product).exists()
    
    # Check if product is in user's cart
    in_cart = False
    cart_quantity = 0
    if request.user.is_authenticated:
        cart_item = Cart.objects.filter(user=request.user, product=product).first()
        if cart_item:
            in_cart = True
            cart_quantity = cart_item.quantity
    
    context = {
        'product': product,
        'related_products': related_products,
        'reviews': reviews,
        'in_wishlist': in_wishlist,
        'in_cart': in_cart,
        'cart_quantity': cart_quantity,
    }
    return render(request, 'bika/pages/product_detail.html', context)

# ADD THESE HELPER FUNCTIONS FOR PRODUCT DETAIL PAGE
@login_required
def add_review(request, product_id):
    """Add product review"""
    if request.method == 'POST':
        product = get_object_or_404(Product, id=product_id)
        rating = request.POST.get('rating')
        comment = request.POST.get('comment')
        
        # Check if user already reviewed this product
        existing_review = ProductReview.objects.filter(
            user=request.user, 
            product=product
        ).first()
        
        if existing_review:
            messages.warning(request, 'You have already reviewed this product!')
        else:
            ProductReview.objects.create(
                user=request.user,
                product=product,
                rating=rating,
                comment=comment,
                is_approved=True  # Auto-approve for now
            )
            messages.success(request, 'Thank you for your review!')
        
        return redirect('bika:product_detail', slug=product.slug)
    
    return redirect('bika:home')

@login_required
def quick_add_to_cart(request, product_id):
    """Quick add to cart from product detail page"""
    if request.method == 'POST':
        product = get_object_or_404(Product, id=product_id)
        quantity = int(request.POST.get('quantity', 1))
        
        if product.stock_quantity < quantity:
            messages.error(request, f'Only {product.stock_quantity} items available!')
            return redirect('bika:product_detail', slug=product.slug)
        
        cart_item, created = Cart.objects.get_or_create(
            user=request.user,
            product=product,
            defaults={'quantity': quantity}
        )
        
        if not created:
            cart_item.quantity += quantity
            cart_item.save()
        
        messages.success(request, f'{product.name} added to cart!')
        return redirect('bika:product_detail', slug=product.slug)
    
    return redirect('bika:home')

@login_required
def quick_add_to_cart(request, product_id):
    """Quick add to cart from product detail page"""
    if request.method == 'POST':
        product = get_object_or_404(Product, id=product_id)
        quantity = int(request.POST.get('quantity', 1))
        
        # Check stock availability
        if product.stock_quantity < quantity:
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
        
        messages.success(request, f'{product.name} added to cart!')
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            cart_count = Cart.objects.filter(user=request.user).count()
            return JsonResponse({
                'success': True,
                'message': f'{product.name} added to cart!',
                'cart_count': cart_count
            })
        
        return redirect('bika:product_detail', slug=product.slug)
    
    return redirect('bika:home')

@login_required
def add_review(request, product_id):
    """Add product review"""
    if request.method == 'POST':
        product = get_object_or_404(Product, id=product_id)
        rating = request.POST.get('rating')
        comment = request.POST.get('comment')
        
        # Validate rating
        if not rating or not rating.isdigit() or int(rating) not in [1, 2, 3, 4, 5]:
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
            ProductReview.objects.create(
                user=request.user,
                product=product,
                rating=int(rating),
                comment=comment,
                is_approved=True  # Auto-approve for now
            )
            messages.success(request, 'Thank you for your review!')
        
        return redirect('bika:product_detail', slug=product.slug)
    
    return redirect('bika:home')
@login_required
def vendor_product_list(request):
    """Vendor's product list with enhanced functionality"""
    if not request.user.is_vendor() and not request.user.is_staff:
        messages.error(request, "Access denied. Vendor account required.")
        return redirect('bika:home')
    
    # Handle bulk actions via POST
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return handle_bulk_actions(request)
    
    # For staff, show all products; for vendors, show only their products
    if request.user.is_staff:
        products = Product.objects.all()
    else:
        products = Product.objects.filter(vendor=request.user)
    
    # Apply filters
    query = request.GET.get('q', '')
    status_filter = request.GET.get('status', '')
    stock_filter = request.GET.get('stock', '')
    
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
    
    # Calculate statistics
    active_products_count = products.filter(status='active').count()
    draft_products_count = products.filter(status='draft').count()
    low_stock_count = products.filter(
        stock_quantity__gt=0, 
        stock_quantity__lte=F('low_stock_threshold')
    ).count()
    out_of_stock_count = products.filter(stock_quantity=0).count()
    
    # Pagination
    paginator = Paginator(products.order_by('-updated_at'), 10)
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.get_page(1)
    except EmptyPage:
        page_obj = paginator.get_page(paginator.num_pages)
    
    context = {
        'products': page_obj,
        'active_products_count': active_products_count,
        'draft_products_count': draft_products_count,
        'low_stock_count': low_stock_count,
        'out_of_stock_count': out_of_stock_count,
        'query': query,
        'status_filter': status_filter,
        'stock_filter': stock_filter,
    }
    return render(request, 'bika/pages/vendor/products.html', context)

def handle_bulk_actions(request):
    """Handle bulk actions for products"""
    try:
        data = request.POST
        action = data.get('action')
        product_ids = json.loads(data.get('product_ids', '[]'))
        
        if not product_ids:
            return JsonResponse({'success': False, 'message': 'No products selected'})
        
        # Get products (respect user permissions)
        if request.user.is_staff:
            products = Product.objects.filter(id__in=product_ids)
        else:
            products = Product.objects.filter(id__in=product_ids, vendor=request.user)
        
        if action == 'activate':
            products.update(status='active')
            return JsonResponse({'success': True, 'message': f'{products.count()} products activated'})
        elif action == 'draft':
            products.update(status='draft')
            return JsonResponse({'success': True, 'message': f'{products.count()} products moved to draft'})
        elif action == 'delete':
            count = products.count()
            products.delete()
            return JsonResponse({'success': True, 'message': f'{count} products deleted'})
        else:
            return JsonResponse({'success': False, 'message': 'Invalid action'})
            
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})
    

from .payment_gateways import AirtelAfricaGateway, PaymentGatewayFactory, StripeGateway
import json

@login_required
def checkout(request):
    """Enhanced checkout with multi-country payment options"""
    cart_items = Cart.objects.filter(user=request.user).select_related('product')
    
    if not cart_items:
        messages.error(request, "Your cart is empty!")
        return redirect('bika:cart')
    
    # Calculate totals
    subtotal = sum(item.total_price for item in cart_items)
    tax_rate = 0.18
    tax_amount = subtotal * tax_rate
    shipping_cost = 5000
    total_amount = subtotal + tax_amount + shipping_cost
    
    # Get user's country (you can get this from user profile or IP)
    user_country = get_user_country(request)
    
    # Get available payment methods for user's country
    available_methods = get_available_payment_methods(user_country)
    
    context = {
        'cart_items': cart_items,
        'subtotal': subtotal,
        'tax_amount': tax_amount,
        'shipping_cost': shipping_cost,
        'total_amount': total_amount,
        'available_methods': available_methods,
        'user_country': user_country,
    }
    return render(request, 'bika/pages/checkout.html', context)

def get_user_country(request):
    """Get user's country from IP or profile"""
    # You can implement IP geolocation here
    # For now, return a default or get from user profile
    return 'TZ'  # Default to Tanzania

def get_available_payment_methods(country_code):
    """Get available payment methods for country"""
    country_methods = {
        'TZ': [
            {'value': 'mpesa', 'name': 'M-Pesa Tanzania', 'icon': 'fas fa-mobile-alt', 'color': 'success'},
            {'value': 'tigo_tz', 'name': 'Tigo Pesa', 'icon': 'fas fa-sim-card', 'color': 'primary'},
            {'value': 'airtel_tz', 'name': 'Airtel Money', 'icon': 'fas fa-wifi', 'color': 'red'},
            {'value': 'visa', 'name': 'Visa Card', 'icon': 'fab fa-cc-visa', 'color': 'navy'},
            {'value': 'mastercard', 'name': 'MasterCard', 'icon': 'fab fa-cc-mastercard', 'color': 'orange'},
            {'value': 'paypal', 'name': 'PayPal', 'icon': 'fab fa-paypal', 'color': 'info'},
        ],
        'RW': [
            {'value': 'mtn_rw', 'name': 'MTN Mobile Money', 'icon': 'fas fa-mobile-alt', 'color': 'yellow'},
            {'value': 'airtel_rw', 'name': 'Airtel Money', 'icon': 'fas fa-wifi', 'color': 'red'},
            {'value': 'visa', 'name': 'Visa Card', 'icon': 'fab fa-cc-visa', 'color': 'navy'},
            {'value': 'mastercard', 'name': 'MasterCard', 'icon': 'fab fa-cc-mastercard', 'color': 'orange'},
            {'value': 'paypal', 'name': 'PayPal', 'icon': 'fab fa-paypal', 'color': 'info'},
        ],
        'UG': [
            {'value': 'mtn_ug', 'name': 'MTN Mobile Money', 'icon': 'fas fa-mobile-alt', 'color': 'yellow'},
            {'value': 'airtel_ug', 'name': 'Airtel Money', 'icon': 'fas fa-wifi', 'color': 'red'},
            {'value': 'visa', 'name': 'Visa Card', 'icon': 'fab fa-cc-visa', 'color': 'navy'},
            {'value': 'mastercard', 'name': 'MasterCard', 'icon': 'fab fa-cc-mastercard', 'color': 'orange'},
            {'value': 'paypal', 'name': 'PayPal', 'icon': 'fab fa-paypal', 'color': 'info'},
        ],
        'KE': [
            {'value': 'mpesa_ke', 'name': 'M-Pesa Kenya', 'icon': 'fas fa-mobile-alt', 'color': 'success'},
            {'value': 'visa', 'name': 'Visa Card', 'icon': 'fab fa-cc-visa', 'color': 'navy'},
            {'value': 'mastercard', 'name': 'MasterCard', 'icon': 'fab fa-cc-mastercard', 'color': 'orange'},
            {'value': 'paypal', 'name': 'PayPal', 'icon': 'fab fa-paypal', 'color': 'info'},
        ],
    }
    
    return country_methods.get(country_code, [
        {'value': 'visa', 'name': 'Visa Card', 'icon': 'fab fa-cc-visa', 'color': 'navy'},
        {'value': 'mastercard', 'name': 'MasterCard', 'icon': 'fab fa-cc-mastercard', 'color': 'orange'},
        {'value': 'paypal', 'name': 'PayPal', 'icon': 'fab fa-paypal', 'color': 'info'},
    ])

@login_required
def initiate_payment(request):
    """Enhanced payment initiation with multi-provider support"""
    if request.method == 'POST':
        try:
            order_id = request.POST.get('order_id')
            payment_method = request.POST.get('payment_method')
            phone_number = request.POST.get('phone_number', '')
            card_token = request.POST.get('card_token', '')
            
            order = get_object_or_404(Order, id=order_id, user=request.user)
            
            # Create payment record
            payment = Payment.objects.create(
                order=order,
                payment_method=payment_method,
                amount=order.total_amount,
                currency='TZS'  # You can make this dynamic based on country
            )
            
            # Get payment gateway configuration
            gateway_config = payment.get_payment_provider_config()
            if not gateway_config:
                return JsonResponse({'success': False, 'message': 'Payment method not available'})
            
            # Create gateway instance
            gateway = PaymentGatewayFactory.create_gateway(payment_method, gateway_config)
            if not gateway:
                return JsonResponse({'success': False, 'message': 'Payment gateway error'})
            
            # Process payment based on method
            if payment_method in ['mpesa', 'tigo_tz', 'airtel_tz', 'mtn_rw', 'airtel_rw', 'mtn_ug', 'airtel_ug', 'mpesa_ke']:
                # Mobile Money payment
                if not phone_number:
                    return JsonResponse({'success': False, 'message': 'Phone number required'})
                
                result = process_mobile_money_payment(gateway, payment_method, phone_number, order.total_amount, order.order_number)
                
            elif payment_method in ['visa', 'mastercard', 'amex']:
                # Card payment
                if not card_token:
                    return JsonResponse({'success': False, 'message': 'Card token required'})
                
                result = process_card_payment(gateway, card_token, order.total_amount, 'TZS', request.user.email)
                
            elif payment_method == 'paypal':
                # PayPal payment
                return_url = request.build_absolute_uri(f'/payment/success/{payment.id}/')
                cancel_url = request.build_absolute_uri(f'/payment/failed/{payment.id}/')
                result = gateway.create_order(order.total_amount, 'USD', return_url, cancel_url)
                
            else:
                return JsonResponse({'success': False, 'message': 'Payment method not supported'})
            
            if result['success']:
                # Update payment record with gateway response
                payment.transaction_id = result.get('transaction_id', '')
                payment.mobile_money_phone = phone_number
                payment.mobile_money_provider = payment_method
                payment.save()
                
                return JsonResponse({
                    'success': True,
                    'payment_id': payment.id,
                    'message': result.get('message', 'Payment initiated successfully'),
                    'approval_url': result.get('approval_url'),  # For PayPal
                    'client_secret': result.get('client_secret'),  # For Stripe
                })
            else:
                payment.status = 'failed'
                payment.save()
                return JsonResponse({'success': False, 'message': result['message']})
                
        except Exception as e:
            logger.error(f"Payment initiation error: {str(e)}")
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request'})

def process_mobile_money_payment(gateway, method, phone_number, amount, reference):
    """Process mobile money payment"""
    country_map = {
        'mpesa': 'TZ', 'tigo_tz': 'TZ', 'airtel_tz': 'TZ',
        'mtn_rw': 'RW', 'airtel_rw': 'RW',
        'mtn_ug': 'UG', 'airtel_ug': 'UG',
        'mpesa_ke': 'KE',
    }
    
    country = country_map.get(method, 'TZ')
    
    if isinstance(gateway, AirtelAfricaGateway):
        return gateway.initiate_payment(phone_number, amount, reference, country)
    else:
        return gateway.initiate_payment(phone_number, amount, reference)

def process_card_payment(gateway, card_token, amount, currency, customer_email):
    """Process card payment"""
    if isinstance(gateway, StripeGateway):
        return gateway.create_payment_intent(amount, currency, card_token, customer_email)
    else:
        return {'success': False, 'message': 'Card payments not configured'}

@csrf_exempt
def stripe_webhook(request):
    """Handle Stripe webhooks"""
    if request.method == 'POST':
        try:
            payload = request.body
            sig_header = request.META['HTTP_STRIPE_SIGNATURE']
            
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
            )
            
            if event['type'] == 'payment_intent.succeeded':
                payment_intent = event['data']['object']
                # Update payment status
                payment = Payment.objects.filter(transaction_id=payment_intent['id']).first()
                if payment:
                    payment.status = 'completed'
                    payment.paid_at = timezone.now()
                    payment.save()
                    
                    payment.order.status = 'confirmed'
                    payment.order.save()
            
            return JsonResponse({'success': True})
            
        except Exception as e:
            logger.error(f"Stripe webhook error: {str(e)}")
            return JsonResponse({'success': False}, status=400)
    
    return JsonResponse({'success': False}, status=405)    

@login_required
def track_my_products(request):
    """Track vendor's products with enhanced analytics"""
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
    status_filter = request.GET.get('status', '')
    sort_filter = request.GET.get('sort', 'updated')
    
    if query:
        products = products.filter(
            Q(name__icontains=query) | 
            Q(sku__icontains=query) |
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
    
    # Apply sorting
    if sort_filter == 'stock':
        products = products.order_by('stock_quantity')
    elif sort_filter == 'name':
        products = products.order_by('name')
    else:  # updated
        products = products.order_by('-updated_at')
    
    # Calculate statistics
    total_products = products.count()
    low_stock_count = products.filter(
        stock_quantity__gt=0, 
        stock_quantity__lte=F('low_stock_threshold')
    ).count()
    out_of_stock_count = products.filter(stock_quantity=0).count()
    active_alerts_count = alerts.count()
    
    # Add sensor data and alert status to products
    for product in products:
        # Check for critical alerts
        product.has_critical_alerts = alerts.filter(
            product=product, 
            severity__in=['critical', 'high']
        ).exists()
        
        # Mock sensor data (in real implementation, fetch from RealTimeSensorData)
        product.sensor_data = [
            {'sensor_type': 'temperature', 'value': 22.5, 'unit': '°C', 'is_normal': True, 'is_warning': False},
            {'sensor_type': 'humidity', 'value': 45, 'unit': '%', 'is_normal': True, 'is_warning': False},
        ]
    
    context = {
        'products': products,
        'alerts': alerts[:10],  # Show latest 10 alerts
        'total_products': total_products,
        'low_stock_count': low_stock_count,
        'out_of_stock_count': out_of_stock_count,
        'active_alerts_count': active_alerts_count,
        'query': query,
        'stock_filter': stock_filter,
        'status_filter': status_filter,
        'sort_filter': sort_filter,
    }
    return render(request, 'bika/pages/vendor/track_products.html', context)

@login_required
def product_analytics_api(request, product_id):
    """API endpoint for product analytics"""
    if request.user.is_staff:
        product = get_object_or_404(Product, id=product_id)
    else:
        product = get_object_or_404(Product, id=product_id, vendor=request.user)
    
    # Mock analytics data - in real implementation, calculate from actual data
    analytics_data = {
        'product_name': product.name,
        'total_sales': 150,  # Calculate from OrderItems
        'total_revenue': float(product.price * 150),
        'current_stock': product.stock_quantity,
        'stockout_count': 3,  # Calculate from history
        'views_count': product.views_count,
        'wishlist_count': Wishlist.objects.filter(product=product).count(),
        'review_count': ProductReview.objects.filter(product=product).count(),
    }
    
    return JsonResponse(analytics_data)

@login_required
def resolve_alert(request, alert_id):
    """Resolve a product alert"""
    if request.method == 'POST':
        try:
            if request.user.is_staff:
                alert = get_object_or_404(ProductAlert, id=alert_id)
            else:
                alert = get_object_or_404(ProductAlert, id=alert_id, product__vendor=request.user)
            
            alert.is_resolved = True
            alert.resolved_at = timezone.now()
            alert.resolved_by = request.user
            alert.save()
            
            return JsonResponse({'success': True, 'message': 'Alert resolved successfully'})
            
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request'})