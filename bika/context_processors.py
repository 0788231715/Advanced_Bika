# bika/context_processors.py
from django.db import DatabaseError
from .models import SiteInfo, Service, Cart, ProductCategory, Product, Notification

def site_info(request):
    """Add comprehensive site information to all templates"""
    context = {}
    
    # 1. Site Information
    try:
        site_info_obj = SiteInfo.objects.first()
        if not site_info_obj:
            # Create default site info if it doesn't exist
            site_info_obj = SiteInfo.objects.create(
                name="Bika",
                tagline="AI-Powered Fruit Quality Monitoring & E-commerce Platform",
                description="Your Success Is Our Business - Bika provides exceptional services to help your business grow.",
                email="contact@bika.com",
                phone="+255 123 456 789",
                address="Dar es Salaam, Tanzania",
                facebook_url="https://facebook.com/bika",
                twitter_url="https://twitter.com/bika",
                instagram_url="https://instagram.com/bika",
                linkedin_url="https://linkedin.com/company/bika",
            )
        context['site_info'] = site_info_obj
        context['site_name'] = site_info_obj.name
        context['site_email'] = site_info_obj.email
        context['site_phone'] = site_info_obj.phone
        context['site_address'] = site_info_obj.address
    except DatabaseError:
        # Database not ready (tables not created yet)
        context['site_info'] = None
        context['site_name'] = 'Bika'
        context['site_email'] = 'contact@bika.com'
        context['site_phone'] = '+255 123 456 789'
        context['site_address'] = 'Dar es Salaam, Tanzania'
    except Exception as e:
        print(f"SiteInfo error: {e}")
        context['site_info'] = None
        context['site_name'] = 'Bika'
        context['site_email'] = 'contact@bika.com'
    
    # 2. Featured Services (for navigation dropdown)
    try:
        context['featured_services'] = Service.objects.filter(
            is_active=True
        ).order_by('display_order')[:6]
    except Exception:
        context['featured_services'] = []
    
    # 3. Product Categories (for navigation)
    try:
        context['product_categories'] = ProductCategory.objects.filter(
            is_active=True,
            parent__isnull=True  # Only top-level categories
        ).order_by('display_order')[:8]
    except Exception:
        context['product_categories'] = []
    
    # 4. Cart Count (for header badge)
    try:
        if request.user.is_authenticated:
            context['cart_count'] = Cart.objects.filter(user=request.user).count()
            context['cart_total'] = sum(
                item.total_price for item in 
                Cart.objects.filter(user=request.user).select_related('product')
            )
        else:
            context['cart_count'] = 0
            context['cart_total'] = 0
    except Exception:
        context['cart_count'] = 0
        context['cart_total'] = 0
    
    # 5. Wishlist Count
    try:
        if request.user.is_authenticated:
            from .models import Wishlist
            context['wishlist_count'] = Wishlist.objects.filter(user=request.user).count()
        else:
            context['wishlist_count'] = 0
    except Exception:
        context['wishlist_count'] = 0
    
    # 6. Unread Notifications Count
    try:
        if request.user.is_authenticated:
            context['unread_notifications_count'] = Notification.objects.filter(
                user=request.user,
                is_read=False
            ).count()
            
            # Get critical notifications
            context['critical_notifications_count'] = Notification.objects.filter(
                user=request.user,
                is_read=False,
                notification_type='urgent_alert'
            ).count()
        else:
            context['unread_notifications_count'] = 0
            context['critical_notifications_count'] = 0
    except Exception:
        context['unread_notifications_count'] = 0
        context['critical_notifications_count'] = 0
    
    # 7. Featured Products (for mini display in header)
    try:
        context['featured_products_header'] = Product.objects.filter(
            status='active',
            is_featured=True
        ).select_related('category', 'vendor')[:3]
    except Exception:
        context['featured_products_header'] = []
    
    # 8. User Type Information
    if request.user.is_authenticated:
        context['user_is_vendor'] = request.user.is_vendor()
        context['user_is_customer'] = request.user.is_customer()
        context['user_is_admin'] = request.user.is_staff or request.user.user_type == 'admin'
        context['user_has_business'] = bool(request.user.business_name)
    else:
        context['user_is_vendor'] = False
        context['user_is_customer'] = False
        context['user_is_admin'] = False
        context['user_has_business'] = False
    
    # 9. Current Year (for footer)
    import datetime
    context['current_year'] = datetime.datetime.now().year
    
    # 10. Development/Production Mode
    from django.conf import settings
    context['debug_mode'] = settings.DEBUG
    
    # 11. Currency Information
    context['default_currency'] = 'TZS'  # Tanzanian Shilling
    context['currency_symbol'] = 'TSh'
    
    # 12. Site Statistics (for footer)
    try:
        context['total_products'] = Product.objects.filter(status='active').count()
        context['total_categories'] = ProductCategory.objects.filter(is_active=True).count()
        context['total_vendors'] = Cart.objects.filter(
            user__user_type='vendor',
            user__is_active=True
        ).values('user').distinct().count()
    except Exception:
        context['total_products'] = 0
        context['total_categories'] = 0
        context['total_vendors'] = 0
    
    # 13. Fruit Monitoring Stats (for vendor/admin dashboard)
    try:
        if request.user.is_authenticated and (request.user.is_vendor() or request.user.is_staff):
            from .models import FruitBatch, FruitQualityReading
            context['fruit_batch_count'] = FruitBatch.objects.filter(
                product__vendor=request.user
            ).count() if request.user.is_vendor() else FruitBatch.objects.count()
            context['fruit_reading_count'] = FruitQualityReading.objects.count()
        else:
            context['fruit_batch_count'] = 0
            context['fruit_reading_count'] = 0
    except Exception:
        context['fruit_batch_count'] = 0
        context['fruit_reading_count'] = 0
    
    # 14. Current Path (for active menu highlighting)
    context['current_path'] = request.path
    
    # 15. Query Parameters (for maintaining filters)
    context['query_params'] = request.GET.urlencode()
    
    # 16. Meta Information
    context['meta_description'] = getattr(site_info_obj, 'meta_description', 
        "Bika is an AI-powered platform for fruit quality monitoring and e-commerce. "
        "We help businesses optimize storage, predict quality, and increase sales.")
    
    context['meta_title'] = getattr(site_info_obj, 'meta_title', 
        "Bika - AI-Powered Fruit Quality Monitoring & E-commerce Platform")
    
    # 17. Social Media URLs
    context['social_facebook'] = getattr(site_info_obj, 'facebook_url', '')
    context['social_twitter'] = getattr(site_info_obj, 'twitter_url', '')
    context['social_instagram'] = getattr(site_info_obj, 'instagram_url', '')
    context['social_linkedin'] = getattr(site_info_obj, 'linkedin_url', '')
    
    return context


def cart_details(request):
    """Detailed cart information (can be used in cart-specific pages)"""
    context = {}
    
    try:
        if request.user.is_authenticated:
            cart_items = Cart.objects.filter(
                user=request.user
            ).select_related('product')
            
            subtotal = sum(item.total_price for item in cart_items)
            tax_rate = 0.18  # 18% VAT
            tax_amount = subtotal * tax_rate
            shipping_cost = 5000  # Fixed shipping cost
            total_amount = subtotal + tax_amount + shipping_cost
            
            context['cart_items_detailed'] = cart_items
            context['cart_subtotal'] = subtotal
            context['cart_tax_amount'] = tax_amount
            context['cart_shipping_cost'] = shipping_cost
            context['cart_total_amount'] = total_amount
            context['cart_tax_rate'] = tax_rate
        else:
            context['cart_items_detailed'] = []
            context['cart_subtotal'] = 0
            context['cart_tax_amount'] = 0
            context['cart_shipping_cost'] = 0
            context['cart_total_amount'] = 0
            context['cart_tax_rate'] = 0.18
    except Exception as e:
        print(f"Cart details error: {e}")
        context['cart_items_detailed'] = []
        context['cart_subtotal'] = 0
        context['cart_tax_amount'] = 0
        context['cart_shipping_cost'] = 0
        context['cart_total_amount'] = 0
        context['cart_tax_rate'] = 0.18
    
    return context


def user_profile_info(request):
    """User profile information"""
    context = {}
    
    if request.user.is_authenticated:
        context['user_profile'] = request.user
        
        # Check if user has any pending actions
        try:
            from .models import Order, ProductAlert
            context['pending_orders'] = Order.objects.filter(
                user=request.user,
                status='pending'
            ).count()
            
            if request.user.is_vendor():
                context['unresolved_alerts'] = ProductAlert.objects.filter(
                    product__vendor=request.user,
                    is_resolved=False
                ).count()
            else:
                context['unresolved_alerts'] = 0
                
        except Exception:
            context['pending_orders'] = 0
            context['unresolved_alerts'] = 0
    
    return context


# List of all context processors
context_processors_list = [
    'bika.context_processors.site_info',
    'bika.context_processors.cart_details',
    'bika.context_processors.user_profile_info',
]