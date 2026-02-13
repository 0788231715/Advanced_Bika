# bika/admin.py - UPDATED WITH NEW MODELS
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from django.urls import path
from django.utils.html import format_html
from django.urls import reverse 
from django.utils import timezone
from django.utils.html import format_html
from django.db.models import Q, F, Count, Sum
from datetime import timedelta
from django.conf import settings
from django.contrib import messages
from django.urls import reverse
from .models import *

# ==================== DASHBOARD VIEW ====================

@staff_member_required
def admin_dashboard(request):
    """Enhanced admin dashboard with comprehensive statistics"""
    now = timezone.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)
    
    # User Statistics
    user_stats = {
        'total': CustomUser.objects.count(),
        'today': CustomUser.objects.filter(date_joined__gte=today_start).count(),
        'yesterday': CustomUser.objects.filter(date_joined__gte=yesterday_start, date_joined__lt=today_start).count(),
        'week': CustomUser.objects.filter(date_joined__gte=week_start).count(),
        'month': CustomUser.objects.filter(date_joined__gte=month_start).count(),
        'admins': CustomUser.objects.filter(user_type='admin').count(),
        'vendors': CustomUser.objects.filter(user_type='vendor', is_active=True).count(),
        'customers': CustomUser.objects.filter(user_type='customer', is_active=True).count(),
        'active': CustomUser.objects.filter(is_active=True).count(),
        'inactive': CustomUser.objects.filter(is_active=False).count(),
    }
    
    # Product Statistics
    product_stats = {
        'total': Product.objects.count(),
        'active': Product.objects.filter(status='active').count(),
        'draft': Product.objects.filter(status='draft').count(),
        'out_of_stock': Product.objects.filter(stock_quantity=0, track_inventory=True).count(),
        'low_stock': Product.objects.filter(
            stock_quantity__gt=0,
            stock_quantity__lte=F('low_stock_threshold'),
            track_inventory=True
        ).count(),
        'featured': Product.objects.filter(is_featured=True, status='active').count(),
        'digital': Product.objects.filter(is_digital=True).count(),
        'today': Product.objects.filter(created_at__gte=today_start).count(),
        'week': Product.objects.filter(created_at__gte=week_start).count(),
    }
    
    # Order Statistics
    order_stats = {
        'total': Order.objects.count(),
        'pending': Order.objects.filter(status='pending').count(),
        'confirmed': Order.objects.filter(status='confirmed').count(),
        'shipped': Order.objects.filter(status='shipped').count(),
        'delivered': Order.objects.filter(status='delivered').count(),
        'cancelled': Order.objects.filter(status='cancelled').count(),
        'today': Order.objects.filter(created_at__gte=today_start).count(),
        'week': Order.objects.filter(created_at__gte=week_start).count(),
        'month': Order.objects.filter(created_at__gte=month_start).count(),
    }
    
    # Calculate revenue
    completed_orders = Order.objects.filter(status='delivered')
    total_revenue = sum(order.total_amount for order in completed_orders if order.total_amount)
    today_revenue = sum(
        order.total_amount for order in completed_orders.filter(
            created_at__gte=today_start
        ) if order.total_amount
    )
    week_revenue = sum(
        order.total_amount for order in completed_orders.filter(
            created_at__gte=week_start
        ) if order.total_amount
    )
    
    # Payment Statistics
    payment_stats = {
        'total': Payment.objects.count(),
        'completed': Payment.objects.filter(status='completed').count(),
        'pending': Payment.objects.filter(status='pending').count(),
        'failed': Payment.objects.filter(status='failed').count(),
        'refunded': Payment.objects.filter(status='refunded').count(),
        'today': Payment.objects.filter(created_at__gte=today_start).count(),
    }
    
    # Category Statistics
    category_stats = {
        'total': ProductCategory.objects.count(),
        'active': ProductCategory.objects.filter(is_active=True).count(),
        'with_products': ProductCategory.objects.filter(
            products__status='active'
        ).distinct().count(),
        'top_categories': ProductCategory.objects.annotate(
            product_count=Count('products', filter=Q(products__status='active'))
        ).order_by('-product_count')[:5],
    }
    
    # Fruit Monitoring Stats
    fruit_stats = {
        'batches': FruitBatch.objects.count(),
        'active_batches': FruitBatch.objects.filter(status='active').count(),
        'completed_batches': FruitBatch.objects.filter(status='completed').count(),
        'fruit_types': FruitType.objects.count(),
        'quality_readings': FruitQualityReading.objects.count(),
        'today_readings': FruitQualityReading.objects.filter(timestamp__gte=today_start).count(),
    }
    
    # Storage Stats
    storage_stats = {
        'locations': StorageLocation.objects.count(),
        'active_locations': StorageLocation.objects.filter(is_active=True).count(),
        'total_capacity': sum(location.capacity for location in StorageLocation.objects.all()),
        'total_occupancy': sum(location.current_occupancy for location in StorageLocation.objects.all()),
        'occupancy_rate': round((sum(location.current_occupancy for location in StorageLocation.objects.all()) / 
                               sum(location.capacity for location in StorageLocation.objects.all()) * 100), 2) 
                               if sum(location.capacity for location in StorageLocation.objects.all()) > 0 else 0,
    }
    
    # Alert Stats
    alert_stats = {
        'total_alerts': ProductAlert.objects.count(),
        'unresolved_alerts': ProductAlert.objects.filter(is_resolved=False).count(),
        'critical_alerts': ProductAlert.objects.filter(severity='critical', is_resolved=False).count(),
        'high_alerts': ProductAlert.objects.filter(severity='high', is_resolved=False).count(),
        'medium_alerts': ProductAlert.objects.filter(severity='medium', is_resolved=False).count(),
        'low_alerts': ProductAlert.objects.filter(severity='low', is_resolved=False).count(),
    }
    
    # INVENTORY STATISTICS (NEW)
    inventory_stats = {
        'total': InventoryItem.objects.count(),
        'active': InventoryItem.objects.filter(status='active').count(),
        'reserved': InventoryItem.objects.filter(status='reserved').count(),
        'sold': InventoryItem.objects.filter(status='sold').count(),
        'low_stock': InventoryItem.objects.filter(status='active').filter(
            quantity__lte=F('low_stock_threshold')
        ).count(),
        'near_expiry': InventoryItem.objects.filter(
            expiry_date__gte=today_start.date(),
            expiry_date__lte=today_start.date() + timedelta(days=30)
        ).count(),
        'expired': InventoryItem.objects.filter(
            expiry_date__lt=today_start.date()
        ).count(),
        'total_value': InventoryItem.objects.aggregate(Sum('total_value'))['total_value__sum'] or 0,
        'by_type': {
            'storage': InventoryItem.objects.filter(item_type='storage').count(),
            'sale': InventoryItem.objects.filter(item_type='sale').count(),
            'rental': InventoryItem.objects.filter(item_type='rental').count(),
        }
    }
    
    # DELIVERY STATISTICS (NEW)
    delivery_stats = {
        'total': Delivery.objects.count(),
        'pending': Delivery.objects.filter(status='pending').count(),
        'processing': Delivery.objects.filter(status='processing').count(),
        'in_transit': Delivery.objects.filter(status='in_transit').count(),
        'delivered': Delivery.objects.filter(status='delivered').count(),
        'cancelled': Delivery.objects.filter(status='cancelled').count(),
        'today': Delivery.objects.filter(created_at__gte=today_start).count(),
        'week': Delivery.objects.filter(created_at__gte=week_start).count(),
        'late': Delivery.objects.filter(
            status__in=['pending', 'processing', 'in_transit', 'out_for_delivery'],
            estimated_delivery__lt=timezone.now()
        ).count(),
        'delivery_rate': round((Delivery.objects.filter(status='delivered').count() / 
                               Delivery.objects.count() * 100), 2) if Delivery.objects.count() > 0 else 0,
    }
    
    # USER ROLE STATISTICS (NEW)
    role_stats = {
        'total_roles': UserRole.objects.count(),
        'by_role': {
            'admin': UserRole.objects.filter(role='admin').count(),
            'manager': UserRole.objects.filter(role='manager').count(),
            'storage_staff': UserRole.objects.filter(role='storage_staff').count(),
            'negotiation_team': UserRole.objects.filter(role='negotiation_team').count(),
            'client': UserRole.objects.filter(role='client').count(),
            'vendor': UserRole.objects.filter(role='vendor').count(),
            'customer': UserRole.objects.filter(role='customer').count(),
        }
    }
    
    # Recent Data
    recent_products = Product.objects.select_related(
        'vendor', 'category'
    ).prefetch_related('images').order_by('-created_at')[:8]
    
    recent_orders = Order.objects.select_related(
        'user'
    ).order_by('-created_at')[:6]
    
    recent_messages = ContactMessage.objects.filter(
        status='new'
    ).order_by('-submitted_at')[:5]
    
    recent_alerts = ProductAlert.objects.filter(
        is_resolved=False
    ).select_related('product').order_by('-created_at')[:5]
    
    # NEW: Recent Inventory Items
    recent_inventory = InventoryItem.objects.select_related(
        'category', 'client', 'location'
    ).order_by('-created_at')[:6]
    
    # NEW: Recent Deliveries
    recent_deliveries = Delivery.objects.select_related(
        'client', 'assigned_to'
    ).order_by('-created_at')[:6]
    
    # NEW: Recent Inventory Changes
    recent_inventory_changes = InventoryHistory.objects.select_related(
        'item', 'user'
    ).order_by('-timestamp')[:8]
    
    # Activity Log (simplified)
    recent_activity = []
    
    # Get Django and Python version
    import django
    import sys
    import platform
    django_version = django.get_version()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_info = f"{platform.system()} {platform.release()}"
    
    # Performance metrics
    total_users = user_stats['total']
    total_products = product_stats['total']
    total_orders = order_stats['total']
    total_inventory = inventory_stats['total']
    total_deliveries = delivery_stats['total']
    
    context = {
        # Statistics
        'user_stats': user_stats,
        'product_stats': product_stats,
        'order_stats': order_stats,
        'payment_stats': payment_stats,
        'category_stats': category_stats,
        'fruit_stats': fruit_stats,
        'storage_stats': storage_stats,
        'alert_stats': alert_stats,
        'inventory_stats': inventory_stats,  # NEW
        'delivery_stats': delivery_stats,    # NEW
        'role_stats': role_stats,           # NEW
        
        # Revenue
        'total_revenue': "{:,.2f}".format(total_revenue) if total_revenue else "0.00",
        'today_revenue': "{:,.2f}".format(today_revenue) if today_revenue else "0.00",
        'week_revenue': "{:,.2f}".format(week_revenue) if week_revenue else "0.00",
        
        # Recent Data
        'recent_products': recent_products,
        'recent_orders': recent_orders,
        'recent_messages': recent_messages,
        'recent_alerts': recent_alerts,
        'recent_inventory': recent_inventory,        # NEW
        'recent_deliveries': recent_deliveries,      # NEW
        'recent_inventory_changes': recent_inventory_changes,  # NEW
        'recent_activity': recent_activity,
        
        # Percentages for charts
        'admin_percentage': round((user_stats['admins'] / user_stats['total'] * 100), 2) if user_stats['total'] > 0 else 0,
        'vendor_percentage': round((user_stats['vendors'] / user_stats['total'] * 100), 2) if user_stats['total'] > 0 else 0,
        'customer_percentage': round((user_stats['customers'] / user_stats['total'] * 100), 2) if user_stats['total'] > 0 else 0,
        'active_products_percentage': round((product_stats['active'] / product_stats['total'] * 100), 2) if product_stats['total'] > 0 else 0,
        'active_users_percentage': round((user_stats['active'] / user_stats['total'] * 100), 2) if user_stats['total'] > 0 else 0,
        'completed_orders_percentage': round((order_stats['delivered'] / order_stats['total'] * 100), 2) if order_stats['total'] > 0 else 0,
        'inventory_active_percentage': round((inventory_stats['active'] / inventory_stats['total'] * 100), 2) if inventory_stats['total'] > 0 else 0,
        'delivery_success_percentage': delivery_stats['delivery_rate'],
        'storage_occupancy_percentage': storage_stats['occupancy_rate'],
        
        # Service stats
        'total_services': Service.objects.count(),
        'total_testimonials': Testimonial.objects.count(),
        'total_messages': ContactMessage.objects.count(),
        'new_messages': ContactMessage.objects.filter(status='new').count(),
        'active_services_count': Service.objects.filter(is_active=True).count(),
        'featured_testimonials_count': Testimonial.objects.filter(is_featured=True, is_active=True).count(),
        'active_faqs_count': FAQ.objects.filter(is_active=True).count(),
        
        # Performance metrics
        'total_users': total_users,
        'total_products': total_products,
        'total_orders': total_orders,
        'total_inventory': total_inventory,
        'total_deliveries': total_deliveries,
        'inventory_value': "{:,.2f}".format(inventory_stats['total_value']) if inventory_stats['total_value'] else "0.00",
        
        # System info
        'django_version': django_version,
        'python_version': python_version,
        'os_info': os_info,
        'debug': settings.DEBUG,
        'now': now,
        'today': today_start.date(),
        'server_time': now.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    return render(request, 'bika/pages/admin/dashboard.html', context)

# ==================== CUSTOM ADMIN ACTIONS ====================

class CustomAdminActions:
    """Custom admin actions for various models"""
    
    @staticmethod
    def mark_as_featured(modeladmin, request, queryset):
        queryset.update(is_featured=True)
        modeladmin.message_user(request, f"{queryset.count()} items marked as featured.")
    
    @staticmethod
    def mark_as_not_featured(modeladmin, request, queryset):
        queryset.update(is_featured=False)
        modeladmin.message_user(request, f"{queryset.count()} items marked as not featured.")
    
    @staticmethod
    def mark_as_active(modeladmin, request, queryset):
        queryset.update(is_active=True)
        modeladmin.message_user(request, f"{queryset.count()} items marked as active.")
    
    @staticmethod
    def mark_as_inactive(modeladmin, request, queryset):
        queryset.update(is_active=False)
        modeladmin.message_user(request, f"{queryset.count()} items marked as inactive.")
    
    @staticmethod
    def mark_as_approved(modeladmin, request, queryset):
        queryset.update(is_approved=True)
        modeladmin.message_user(request, f"{queryset.count()} items marked as approved.")
    
    @staticmethod
    def mark_as_resolved(modeladmin, request, queryset):
        queryset.update(is_resolved=True, resolved_at=timezone.now(), resolved_by=request.user)
        modeladmin.message_user(request, f"{queryset.count()} alerts marked as resolved.")
    
    @staticmethod
    def mark_as_read(modeladmin, request, queryset):
        queryset.update(is_read=True)
        modeladmin.message_user(request, f"{queryset.count()} notifications marked as read.")

# ==================== ADMIN MODEL REGISTRATIONS ====================

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'user_type', 'is_active', 'is_staff', 'date_joined', 'action_buttons']
    list_filter = ['user_type', 'is_active', 'is_staff', 'date_joined']
    search_fields = ['username', 'email', 'first_name', 'last_name', 'business_name']
    readonly_fields = ['date_joined', 'last_login']
    fieldsets = (
        ('Personal Info', {
            'fields': ('username', 'email', 'first_name', 'last_name', 'phone', 'profile_picture')
        }),
        ('User Type & Permissions', {
            'fields': ('user_type', 'is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')
        }),
        ('Vendor Info', {
            'fields': ('business_name', 'business_description', 'business_logo', 'business_verified'),
            'classes': ('collapse',),
        }),
        ('Address', {
            'fields': ('company', 'address'),
            'classes': ('collapse',),
        }),
        ('Verification', {
            'fields': ('email_verified', 'phone_verified'),
            'classes': ('collapse',),
        }),
        ('Important Dates', {
            'fields': ('last_login', 'date_joined'),
            'classes': ('collapse',),
        }),
    )
    actions = ['activate_users', 'deactivate_users', 'make_vendors', 'make_customers']
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_customuser_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def activate_users(self, request, queryset):
        queryset.update(is_active=True)
        self.message_user(request, f"{queryset.count()} users activated.")
    activate_users.short_description = "Activate selected users"
    
    def deactivate_users(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f"{queryset.count()} users deactivated.")
    deactivate_users.short_description = "Deactivate selected users"
    
    def make_vendors(self, request, queryset):
        queryset.update(user_type='vendor')
        self.message_user(request, f"{queryset.count()} users converted to vendors.")
    make_vendors.short_description = "Convert to vendors"
    
    def make_customers(self, request, queryset):
        queryset.update(user_type='customer')
        self.message_user(request, f"{queryset.count()} users converted to customers.")
    make_customers.short_description = "Convert to customers"

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'sku', 'category', 'vendor', 'price', 'stock_status', 
                   'status', 'is_featured', 'created_at', 'action_buttons']
    list_filter = ['status', 'category', 'vendor', 'is_featured', 'is_digital', 'created_at']
    search_fields = ['name', 'sku', 'description', 'short_description', 'tags']
    readonly_fields = ['created_at', 'updated_at', 'published_at', 'views_count']
    list_editable = ['status', 'is_featured']  # These are in list_display
    list_per_page = 20
    actions = ['activate_products', 'draft_products', 'mark_featured', 'unmark_featured']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'slug', 'sku', 'barcode', 'category', 'vendor')
        }),
        ('Descriptions', {
            'fields': ('description', 'short_description', 'tags')
        }),
        ('Pricing', {
            'fields': ('price', 'compare_price', 'cost_price', 'tax_rate')
        }),
        ('Inventory', {
            'fields': ('stock_quantity', 'low_stock_threshold', 'track_inventory', 'allow_backorders')
        }),
        ('Product Details', {
            'fields': ('brand', 'model', 'weight', 'dimensions', 'color', 'size', 'material')
        }),
        ('Status & Visibility', {
            'fields': ('status', 'condition', 'is_featured', 'is_digital')
        }),
        ('SEO', {
            'fields': ('meta_title', 'meta_description'),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'published_at', 'views_count'),
            'classes': ('collapse',),
        }),
    )
    
    def stock_status(self, obj):
        if not obj.track_inventory:
            return format_html('<span class="badge badge-info">Not Tracked</span>')
        if obj.stock_quantity <= 0:
            return format_html('<span class="badge badge-danger">Out of Stock</span>')
        elif obj.stock_quantity <= obj.low_stock_threshold:
            return format_html('<span class="badge badge-warning">Low Stock</span>')
        else:
            return format_html('<span class="badge badge-success">In Stock</span>')
    stock_status.short_description = 'Stock'
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_product_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def activate_products(self, request, queryset):
        queryset.update(status='active')
        self.message_user(request, f"{queryset.count()} products activated.")
    activate_products.short_description = "Activate selected products"
    
    def draft_products(self, request, queryset):
        updated = queryset.update(status='draft')
        self.message_user(request, f"{updated} products moved to draft.")
    draft_products.short_description = "Move to draft"
    
    def mark_featured(self, request, queryset):
        updated = queryset.update(is_featured=True)
        self.message_user(request, f"{updated} products marked as featured.")
    mark_featured.short_description = "Mark as featured"
    
    def unmark_featured(self, request, queryset):
        updated = queryset.update(is_featured=False)
        self.message_user(request, f"{updated} products unmarked as featured.")
    unmark_featured.short_description = "Remove featured status"

@admin.register(ProductCategory)
class ProductCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'product_count', 'is_active', 'display_order']
    list_filter = ['is_active', 'parent']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    list_editable = ['display_order', 'is_active']  # These are in list_display
    
    def product_count(self, obj):
        return obj.products.count()
    product_count.short_description = 'Products'

@admin.register(ProductImage)
class ProductImageAdmin(admin.ModelAdmin):
    list_display = ['product', 'image_preview', 'alt_text', 'display_order', 'is_primary']
    list_filter = ['is_primary', 'product']
    search_fields = ['product__name', 'alt_text']
    list_editable = ['display_order', 'is_primary']  # These are in list_display
    
    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="50" height="50" />', obj.image.url)
        return "-"
    image_preview.short_description = 'Preview'

@admin.register(ProductReview)
class ProductReviewAdmin(admin.ModelAdmin):
    list_display = ['product', 'user', 'rating_stars', 'title', 'is_approved', 
                   'is_verified_purchase', 'created_at']
    list_filter = ['rating', 'is_approved', 'is_verified_purchase', 'created_at']
    search_fields = ['product__name', 'user__username', 'title', 'comment']
    list_editable = ['is_approved']  # This is in list_display
    readonly_fields = ['created_at', 'updated_at']
    actions = ['approve_reviews', 'disapprove_reviews']
    
    def rating_stars(self, obj):
        stars = '★' * obj.rating + '☆' * (5 - obj.rating)
        return format_html('<span style="color: gold; font-size: 14px;">{}</span>', stars)
    rating_stars.short_description = 'Rating'
    
    def approve_reviews(self, request, queryset):
        queryset.update(is_approved=True)
        self.message_user(request, f"{queryset.count()} reviews approved.")
    approve_reviews.short_description = "Approve selected reviews"
    
    def disapprove_reviews(self, request, queryset):
        queryset.update(is_approved=False)
        self.message_user(request, f"{queryset.count()} reviews disapproved.")
    disapprove_reviews.short_description = "Disapprove selected reviews"

# ==================== E-COMMERCE MODELS ====================

@admin.register(Wishlist)
class WishlistAdmin(admin.ModelAdmin):
    list_display = ['user', 'product', 'added_at']
    list_filter = ['added_at']
    search_fields = ['user__username', 'product__name']
    readonly_fields = ['added_at']

@admin.register(Cart)
class CartAdmin(admin.ModelAdmin):
    list_display = ['user', 'product', 'quantity', 'total_price', 'added_at']
    list_filter = ['added_at']
    search_fields = ['user__username', 'product__name']
    readonly_fields = ['added_at', 'updated_at']
    
    def total_price(self, obj):
        return f"${obj.total_price:.2f}" if obj.total_price else "$0.00"
    total_price.short_description = 'Total'

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['order_number', 'user', 'total_amount', 'status', 'created_at', 'action_buttons']
    list_filter = ['status', 'created_at']
    search_fields = ['order_number', 'user__username', 'shipping_address', 'billing_address']
    readonly_fields = ['created_at', 'updated_at', 'order_number']
    list_editable = ['status']  # This is in list_display
    actions = ['confirm_orders', 'ship_orders', 'deliver_orders', 'cancel_orders']
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_order_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def confirm_orders(self, request, queryset):
        queryset.update(status='confirmed')
        self.message_user(request, f"{queryset.count()} orders confirmed.")
    confirm_orders.short_description = "Confirm selected orders"
    
    def ship_orders(self, request, queryset):
        queryset.update(status='shipped')
        self.message_user(request, f"{queryset.count()} orders marked as shipped.")
    ship_orders.short_description = "Mark as shipped"
    
    def deliver_orders(self, request, queryset):
        queryset.update(status='delivered')
        self.message_user(request, f"{queryset.count()} orders marked as delivered.")
    deliver_orders.short_description = "Mark as delivered"
    
    def cancel_orders(self, request, queryset):
        queryset.update(status='cancelled')
        self.message_user(request, f"{queryset.count()} orders cancelled.")
    cancel_orders.short_description = "Cancel selected orders"

@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ['order', 'product', 'quantity', 'price', 'total_price']
    search_fields = ['order__order_number', 'product__name']
    
    def total_price(self, obj):
        return f"${obj.total_price:.2f}" if obj.total_price else "$0.00"
    total_price.short_description = 'Total'

# ==================== PAYMENT MODELS ====================

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['order', 'payment_method_display', 'amount', 'currency', 'status', 'created_at']
    list_filter = ['status', 'payment_method', 'currency', 'created_at']
    search_fields = ['order__order_number', 'transaction_id', 'mobile_money_phone']
    readonly_fields = ['created_at', 'updated_at', 'paid_at']
    
    def payment_method_display(self, obj):
        return obj.get_payment_method_display()
    payment_method_display.short_description = 'Payment Method'

@admin.register(PaymentGatewaySettings)
class PaymentGatewaySettingsAdmin(admin.ModelAdmin):
    list_display = ['gateway_display', 'is_active', 'environment', 'display_name', 'api_user']
    list_filter = ['is_active', 'environment', 'gateway']
    search_fields = ['display_name', 'gateway', 'api_user', 'merchant_id']
    readonly_fields = ['updated_at']
    fieldsets = (
        (None, {
            'fields': ('gateway', 'display_name', 'is_active', 'environment', 'supported_countries', 'supported_currencies')
        }),
        ('API Credentials', {
            'fields': ('api_key', 'api_secret', 'merchant_id', 'webhook_secret', 'api_user', 'api_password'),
            'description': 'Enter credentials provided by the payment gateway. Some fields may be specific to certain gateways.'
        }),
        ('Configuration & Fees', {
            'fields': ('base_url', 'callback_url', 'transaction_fee_percent', 'transaction_fee_fixed')
        }),
        ('Timestamps', {
            'fields': ('updated_at',)
        }),
    )
    
    def gateway_display(self, obj):
        return obj.get_gateway_display()
    gateway_display.short_description = 'Gateway'

@admin.register(CurrencyExchangeRate)
class CurrencyExchangeRateAdmin(admin.ModelAdmin):
    list_display = ['base_currency', 'target_currency', 'exchange_rate', 'last_updated']
    list_filter = ['base_currency', 'target_currency']
    search_fields = ['base_currency', 'target_currency']
    readonly_fields = ['last_updated']

@admin.register(ShippingMethod)
class ShippingMethodAdmin(admin.ModelAdmin):
    list_display = ['name', 'base_cost', 'is_active', 'estimated_delivery_min_days', 'estimated_delivery_max_days', 'updated_at']
    list_filter = ['is_active', 'base_cost']
    search_fields = ['name', 'description']
    list_editable = ['is_active', 'base_cost', 'estimated_delivery_min_days', 'estimated_delivery_max_days']
    readonly_fields = ['created_at', 'updated_at']
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'is_active')
        }),
        ('Pricing & Delivery', {
            'fields': ('base_cost', 'estimated_delivery_min_days', 'estimated_delivery_max_days')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

# ==================== SUPPLIER & PURCHASE ORDER ADMIN ====================

@admin.register(Supplier)
class SupplierAdmin(admin.ModelAdmin):
    list_display = ['name', 'contact_person', 'email', 'phone', 'city', 'country', 'updated_at']
    search_fields = ['name', 'contact_person', 'email', 'phone']
    list_filter = ['country', 'city']
    readonly_fields = ['created_at', 'updated_at']
    fieldsets = (
        (None, {
            'fields': ('name', 'contact_person', 'email', 'phone', 'payment_terms', 'notes')
        }),
        ('Address Information', {
            'fields': ('address', 'city', 'country')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

class PurchaseOrderItemInline(admin.TabularInline):
    model = PurchaseOrderItem
    extra = 1 # Number of empty forms to display

@admin.register(PurchaseOrder)
class PurchaseOrderAdmin(admin.ModelAdmin):
    list_display = ['order_number', 'supplier', 'ordered_by', 'order_date', 'expected_delivery_date', 'status', 'total_amount']
    list_filter = ['status', 'supplier', 'order_date']
    search_fields = ['order_number', 'supplier__name', 'ordered_by__username']
    date_hierarchy = 'order_date'
    inlines = [PurchaseOrderItemInline]
    readonly_fields = ['order_number', 'created_at', 'updated_at'] # order_number should be set automatically
    fieldsets = (
        (None, {
            'fields': ('supplier', 'ordered_by', 'status', 'notes')
        }),
        ('Dates', {
            'fields': ('expected_delivery_date', 'actual_delivery_date', 'order_date'),
            'classes': ('collapse',),
        }),
        ('Totals', {
            'fields': ('total_amount',) # total_amount can be updated based on items
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    def save_formset(self, request, form, formset, change):
        instances = formset.save(commit=False)
        total_amount = Decimal('0.00')
        for instance in instances:
            if instance.product and instance.quantity and instance.unit_price:
                total_amount += instance.total_price
            instance.save()
        form.instance.total_amount = total_amount
        form.instance.save()
        formset.save_m2m() # Required if you have ManyToMany fields in inlines
    
    # Override save_model to ensure total_amount is calculated on save
    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        # Recalculate total amount after items are saved/changed
        obj.total_amount = sum(item.total_price for item in obj.items.all())
        obj.save(update_fields=['total_amount'])


@admin.register(PurchaseOrderItem)
class PurchaseOrderItemAdmin(admin.ModelAdmin):
    list_display = ['purchase_order', 'product', 'quantity', 'unit_price', 'total_price', 'status', 'updated_at']
    list_filter = ['status', 'product__category', 'purchase_order__supplier']
    search_fields = ['product__name', 'purchase_order__order_number']
    readonly_fields = ['created_at', 'updated_at', 'total_price']
    list_editable = ['quantity', 'unit_price', 'status']

@admin.register(InventoryTransfer)
class InventoryTransferAdmin(admin.ModelAdmin):
    list_display = ['transfer_number', 'product', 'quantity', 'source_location', 
                   'destination_location', 'status', 'status_badge', 'requested_by', 'transfer_date'] # Added 'status'
    list_filter = ['status', 'source_location', 'destination_location', 'transfer_date']
    search_fields = ['transfer_number', 'product__name', 'requested_by__username']
    date_hierarchy = 'transfer_date'
    readonly_fields = ['transfer_number', 'transfer_date', 'shipped_date', 'received_date']
    list_editable = ['status'] # Allow quick status updates
    
    fieldsets = (
        ('Transfer Details', {
            'fields': ('transfer_number', 'product', 'quantity', 'status', 'notes')
        }),
        ('Locations', {
            'fields': ('source_location', 'destination_location')
        }),
        ('Personnel & Dates', {
            'fields': ('requested_by', 'approved_by', 'transfer_date', 'shipped_date', 'received_date'),
        }),
    )

    def status_badge(self, obj):
        colors = {
            'pending': 'secondary',
            'in_transit': 'warning',
            'received': 'success',
            'cancelled': 'danger',
            'failed': 'dark',
        }
        color = colors.get(obj.status, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    # Custom actions
    actions = ['mark_as_in_transit', 'mark_as_received', 'mark_as_cancelled']

    def mark_as_in_transit(self, request, queryset):
        count = 0
        for transfer in queryset:
            if transfer.mark_as_shipped(request.user):
                count += 1
        self.message_user(request, f"{count} transfers marked as in transit.")
    mark_as_in_transit.short_description = "Mark selected transfers as In Transit"

    def mark_as_received(self, request, queryset):
        count = 0
        for transfer in queryset:
            if transfer.mark_as_received(request.user):
                count += 1
        self.message_user(request, f"{count} transfers marked as received.")
    mark_as_received.short_description = "Mark selected transfers as Received"

    def mark_as_cancelled(self, request, queryset):
        updated = queryset.update(status='cancelled')
        self.message_user(request, f"{updated} transfers marked as cancelled.")
    mark_as_cancelled.short_description = "Mark selected transfers as Cancelled"

@admin.register(ProductPriceHistory)
class ProductPriceHistoryAdmin(admin.ModelAdmin):
    list_display = ['product', 'old_price', 'new_price', 'change_date', 'changed_by', 'reason']
    list_filter = ['change_date', 'changed_by', 'reason']
    search_fields = ['product__name', 'product__sku', 'reason']
    readonly_fields = ['change_date', 'old_price', 'new_price', 'product', 'changed_by'] # All fields should be read-only after creation
    list_per_page = 20

@admin.register(InventoryMovement)
class InventoryMovementAdmin(admin.ModelAdmin):
    list_display = ['item', 'movement_type', 'from_location', 'to_location', 
                   'quantity_moved', 'moved_by', 'movement_date']
    list_filter = ['movement_type', 'from_location', 'to_location', 'movement_date']
    search_fields = ['item__name', 'item__sku', 'moved_by__username', 'notes']
    date_hierarchy = 'movement_date'
    readonly_fields = ['movement_date']
    list_per_page = 20

# ==================== FRUIT MONITORING MODELS ====================

@admin.register(FruitType)
class FruitTypeAdmin(admin.ModelAdmin):
    list_display = ['name', 'scientific_name', 'optimal_temp_range', 'optimal_humidity_range', 
                   'shelf_life_days', 'batch_count']
    list_filter = ['ethylene_sensitive', 'chilling_sensitive']
    search_fields = ['name', 'scientific_name']
    list_editable = ['shelf_life_days']  # This is in list_display
    
    def optimal_temp_range(self, obj):
        return f"{obj.optimal_temp_min} - {obj.optimal_temp_max}°C"
    optimal_temp_range.short_description = 'Temperature Range'
    
    def optimal_humidity_range(self, obj):
        return f"{obj.optimal_humidity_min} - {obj.optimal_humidity_max}%"
    optimal_humidity_range.short_description = 'Humidity Range'
    
    def batch_count(self, obj):
        return obj.fruitbatch_set.count()
    batch_count.short_description = 'Batches'

@admin.register(FruitBatch)
class FruitBatchAdmin(admin.ModelAdmin):
    list_display = ['batch_number', 'fruit_type', 'quantity', 'arrival_date', 
                   'expected_expiry', 'days_remaining', 'status', 'current_quality']
    list_filter = ['status', 'fruit_type', 'arrival_date', 'storage_location']
    search_fields = ['batch_number', 'fruit_type__name', 'supplier']
    readonly_fields = ['created_at', 'updated_at']
    list_editable = ['status']  # This is in list_display
    
    def days_remaining(self, obj):
        return obj.days_remaining if hasattr(obj, 'days_remaining') else 0
    days_remaining.short_description = 'Days Remaining'
    
    def current_quality(self, obj):
        latest = FruitQualityReading.objects.filter(fruit_batch=obj).order_by('-timestamp').first()
        if latest:
            color = {
                'Fresh': 'success',
                'Good': 'info',
                'Fair': 'warning',
                'Poor': 'danger',
                'Rotten': 'dark',
            }.get(latest.predicted_class, 'secondary')
            return format_html(
                '<span class="badge badge-{}">{}</span>',
                color, latest.predicted_class
            )
        return '-'
    current_quality.short_description = 'Current Quality'

@admin.register(FruitQualityReading)
class FruitQualityReadingAdmin(admin.ModelAdmin):
    list_display = ['fruit_batch', 'timestamp', 'temperature', 'humidity', 
                   'predicted_class_badge', 'confidence_score', 'is_within_optimal_range']
    list_filter = ['predicted_class', 'timestamp', 'fruit_batch__fruit_type']
    search_fields = ['fruit_batch__batch_number', 'notes']
    readonly_fields = ['timestamp']
    list_per_page = 20
    
    def predicted_class_badge(self, obj):
        color = {
            'Fresh': 'success',
            'Good': 'info',
            'Fair': 'warning',
            'Poor': 'danger',
            'Rotten': 'dark',
        }.get(obj.predicted_class, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.predicted_class
        )
    predicted_class_badge.short_description = 'Predicted Quality'
    
    def is_within_optimal_range(self, obj):
        return obj.is_within_optimal_range if hasattr(obj, 'is_within_optimal_range') else False
    is_within_optimal_range.boolean = True
    is_within_optimal_range.short_description = 'Optimal Range'

# ==================== STORAGE & SENSOR MODELS ====================

@admin.register(StorageLocation)
class StorageLocationAdmin(admin.ModelAdmin):
    list_display = ['name', 'address_short', 'capacity', 'current_occupancy', 
                   'available_capacity', 'occupancy_percentage', 'is_active']
    list_filter = ['is_active']
    search_fields = ['name', 'address']
    list_editable = ['is_active']
    
    def address_short(self, obj):
        if len(obj.address) > 30:
            return obj.address[:27] + '...'
        return obj.address
    address_short.short_description = 'Address'
    
    def available_capacity(self, obj):
        # Use the model's property directly
        return obj.available_capacity
    available_capacity.short_description = 'Available'
    
    def occupancy_percentage(self, obj):
        if obj.capacity > 0:
            # Calculate percentage
            percentage = (obj.current_occupancy / obj.capacity) * 100
            
            # Format the percentage to 1 decimal place
            percentage_formatted = f"{percentage:.1f}%"
            
            # Determine color based on occupancy
            if percentage < 80:
                color = 'success'
            elif percentage < 95:
                color = 'warning'
            else:
                color = 'danger'
            
            # Return formatted HTML with progress bar
            return format_html(
                '<div class="progress" style="height: 20px; width: 100px;">'
                '<div class="progress-bar bg-{}" role="progressbar" '
                'style="width: {}%;" aria-valuenow="{}" aria-valuemin="0" '
                'aria-valuemax="100">{}</div></div>',
                color, percentage, percentage, percentage_formatted
            )
        return '-'
    occupancy_percentage.short_description = 'Occupancy'
    
    # Optional: Add a custom change form to show occupancy info
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'address', 'latitude', 'longitude', 'is_active')
        }),
        ('Capacity Information', {
            'fields': ('capacity', 'current_occupancy'),
            'description': 'Set the total capacity and current occupancy of this storage location.'
        }),
    )
    
    # Optional: Add custom help text for capacity fields
    def formfield_for_dbfield(self, db_field, request, **kwargs):
        formfield = super().formfield_for_dbfield(db_field, request, **kwargs)
        if db_field.name == 'capacity':
            formfield.help_text = 'Maximum number of items this storage location can hold.'
        elif db_field.name == 'current_occupancy':
            formfield.help_text = 'Current number of items stored in this location.'
        return formfield

@admin.register(RealTimeSensorData)
class RealTimeSensorDataAdmin(admin.ModelAdmin):
    list_display = ['product', 'fruit_batch', 'sensor_type', 'value_with_unit', 
                   'location', 'recorded_at']
    list_filter = ['sensor_type', 'location', 'recorded_at']
    search_fields = ['product__name', 'fruit_batch__batch_number']
    readonly_fields = ['recorded_at']
    
    def value_with_unit(self, obj):
        return f"{obj.value} {obj.unit}" if obj.unit else str(obj.value)
    value_with_unit.short_description = 'Value'

# ==================== AI & DATASET MODELS ====================

@admin.register(ProductDataset)
class ProductDatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset_type_display', 'row_count', 'is_active', 'created_at']
    list_filter = ['dataset_type', 'is_active', 'created_at']
    search_fields = ['name', 'description']
    
    def dataset_type_display(self, obj):
        return obj.get_dataset_type_display()
    dataset_type_display.short_description = 'Type'

@admin.register(ProductAIInsights)

class ProductAIInsightsAdmin(admin.ModelAdmin):

    list_display = [

        'product', 'predicted_stock_level', 'predicted_out_of_stock_date', 

        'predicted_expiry_date', 'predicted_quality_class_badge', 

        'demand_forecast_next_7_days', 'prediction_confidence', 'last_analyzed', 'analysis_model_used'

    ]

    list_filter = ['predicted_quality_class', 'analysis_model_used__name', 'last_analyzed']

    search_fields = [

        'product__name', 'product__sku', 'predicted_quality_class', 

        'analysis_model_used__name'

    ]

    readonly_fields = ['last_analyzed']

    list_per_page = 20



    def predicted_quality_class_badge(self, obj):

        colors = {

            'Excellent': 'success',

            'Good': 'info',

            'Fair': 'warning',

            'Poor': 'danger',

            'Critical': 'dark',

        }

        color = colors.get(obj.predicted_quality_class, 'secondary')

        return format_html(

            '<span class="badge badge-{}">{}</span>',

            color, obj.predicted_quality_class

        )

    predicted_quality_class_badge.short_description = 'Predicted Quality'



    fieldsets = (

        ('Product Linkage', {

            'fields': ('product', 'inventory_item', 'storage_location'),

            'description': 'Link this insight to a general product, a specific inventory item, or a product in a location.'

        }),

        ('Stock Predictions', {

            'fields': ('predicted_stock_level', 'predicted_out_of_stock_date'),

        }),

        ('Expiry & Quality Predictions', {

            'fields': ('predicted_expiry_date', 'predicted_quality_class'),

        }),

        ('Demand Forecasting', {

            'fields': ('demand_forecast_next_7_days', 'demand_forecast_next_30_days'),

        }),

        ('AI Analysis Details', {

            'fields': ('prediction_confidence', 'last_analyzed', 'analysis_model_used'),

            'classes': ('collapse',),

        }),

    )



@admin.register(AIPredictionAccuracy)

class AIPredictionAccuracyAdmin(admin.ModelAdmin):

    list_display = ['model', 'prediction_type', 'metric_name', 'metric_value', 'evaluation_date', 'product']

    list_filter = ['prediction_type', 'model', 'metric_name', 'evaluation_date']

    search_fields = ['product__name', 'model__name', 'notes']

    readonly_fields = ['evaluation_date', 'period_start', 'period_end']

    list_per_page = 20



@admin.register(TrainedModel)

class TrainedModelAdmin(admin.ModelAdmin):

    list_display = ['name', 'model_type_display', 'dataset', 'accuracy_percentage', 

                   'training_date', 'is_active']

    list_filter = ['model_type', 'is_active', 'training_date']

    search_fields = ['name', 'dataset__name']
    
    def model_type_display(self, obj):
        return obj.get_model_type_display()
    model_type_display.short_description = 'Model Type'
    
    def accuracy_percentage(self, obj):
        if obj.accuracy:
            return f"{obj.accuracy * 100:.2f}%"
        return '-'
    accuracy_percentage.short_description = 'Accuracy'

# ==================== ALERT & NOTIFICATION MODELS ====================

@admin.register(ProductAlert)
class ProductAlertAdmin(admin.ModelAdmin):
    list_display = ['get_product_or_item', 'alert_type_display', 'severity_badge', 'is_resolved', 
                   'created_at', 'resolved_at']
    list_filter = ['alert_type', 'severity', 'is_resolved', 'created_at', 'product', 'inventory_item__location']
    search_fields = ['product__name', 'inventory_item__name', 'message']
    readonly_fields = ['created_at', 'resolved_at']
    list_editable = ['is_resolved']
    actions = ['mark_resolved', 'mark_unresolved']

    def get_product_or_item(self, obj):
        if obj.inventory_item:
            return format_html(
                'Inventory Item: <a href="{}">{}</a> (Location: {})',
                reverse('admin:bika_inventoryitem_change', args=[obj.inventory_item.id]),
                obj.inventory_item.name,
                obj.inventory_item.location.name if obj.inventory_item.location else 'N/A'
            )
        elif obj.product:
            return format_html(
                'Product: <a href="{}">{}</a>',
                reverse('admin:bika_product_change', args=[obj.product.id]),
                obj.product.name
            )
        return 'N/A'
    get_product_or_item.short_description = 'Alert Target'
    
    def alert_type_display(self, obj):
        return obj.get_alert_type_display()
    alert_type_display.short_description = 'Alert Type'
    
    def severity_badge(self, obj):
        colors = {
            'low': 'info',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'dark',
        }
        color = colors.get(obj.severity, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_severity_display()
        )
    severity_badge.short_description = 'Severity'
    
    def mark_resolved(self, request, queryset):
        updated = queryset.update(is_resolved=True, resolved_at=timezone.now(), resolved_by=request.user)
        self.message_user(request, f"{updated} alerts marked as resolved.")
    mark_resolved.short_description = "Mark as resolved"
    
    def mark_unresolved(self, request, queryset):
        updated = queryset.update(is_resolved=False, resolved_at=None, resolved_by=None)
        self.message_user(request, f"{updated} alerts marked as unresolved.")
    mark_unresolved.short_description = "Mark as unresolved"
@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['user', 'title', 'notification_type_display', 'is_read', 
                   'created_at', 'action_buttons']
    list_filter = ['notification_type', 'is_read', 'created_at']
    search_fields = ['user__username', 'title', 'message']
    readonly_fields = ['created_at']
    list_editable = ['is_read']  # This is in list_display
    actions = ['mark_read', 'mark_unread']
    
    def notification_type_display(self, obj):
        return obj.get_notification_type_display()
    notification_type_display.short_description = 'Type'
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_notification_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def mark_read(self, request, queryset):
        updated = queryset.update(is_read=True)
        self.message_user(request, f"{updated} notifications marked as read.")
    mark_read.short_description = "Mark as read"
    
    def mark_unread(self, request, queryset):
        updated = queryset.update(is_read=False)
        self.message_user(request, f"{updated} notifications marked as unread.")
    mark_unread.short_description = "Mark as unread"

# ==================== SITE CONTENT MODELS ====================

@admin.register(SiteInfo)
class SiteInfoAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone', 'updated_at']
    readonly_fields = ['updated_at']
    
    def has_add_permission(self, request):
        # Allow only one instance
        if self.model.objects.count() >= 1:
            return False
        return super().has_add_permission(request)

@admin.register(Service)
class ServiceAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'display_order', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    list_editable = ['display_order', 'is_active']  # These are in list_display

@admin.register(Testimonial)
class TestimonialAdmin(admin.ModelAdmin):
    list_display = ['name', 'company', 'rating_stars', 'is_featured', 'is_active', 'created_at']
    list_filter = ['is_featured', 'is_active', 'rating', 'created_at']
    search_fields = ['name', 'company', 'content']
    list_editable = ['is_featured', 'is_active']  # These are in list_display
    
    def rating_stars(self, obj):
        stars = '★' * obj.rating + '☆' * (5 - obj.rating)
        return format_html('<span style="color: gold; font-size: 14px;">{}</span>', stars)
    rating_stars.short_description = 'Rating'

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'status', 'submitted_at', 'action_buttons']
    list_filter = ['status', 'submitted_at']
    search_fields = ['name', 'email', 'subject', 'message']
    readonly_fields = ['submitted_at', 'ip_address', 'replied_at']
    list_editable = ['status']  # This is in list_display
    actions = ['mark_as_replied', 'mark_as_read', 'mark_as_closed']
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_contactmessage_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def mark_as_replied(self, request, queryset):
        for message in queryset:
            message.mark_as_replied()
        self.message_user(request, f"{queryset.count()} messages marked as replied.")
    mark_as_replied.short_description = "Mark as replied"
    
    def mark_as_read(self, request, queryset):
        queryset.update(status='read')
        self.message_user(request, f"{queryset.count()} messages marked as read.")
    mark_as_read.short_description = "Mark as read"
    
    def mark_as_closed(self, request, queryset):
        queryset.update(status='closed')
        self.message_user(request, f"{queryset.count()} messages marked as closed.")
    mark_as_closed.short_description = "Mark as closed"

@admin.register(FAQ)
class FAQAdmin(admin.ModelAdmin):
    list_display = ['question_short', 'answer_short', 'display_order', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['question', 'answer']
    list_editable = ['display_order', 'is_active']  # These are in list_display
    
    def question_short(self, obj):
        if len(obj.question) > 50:
            return obj.question[:47] + '...'
        return obj.question
    question_short.short_description = 'Question'
    
    def answer_short(self, obj):
        if len(obj.answer) > 50:
            return obj.answer[:47] + '...'
        return obj.answer
    answer_short.short_description = 'Answer'

# ==================== NEW MODELS ADMIN REGISTRATIONS ====================

@admin.register(UserRole)
class UserRoleAdmin(admin.ModelAdmin):
    list_display = ['user', 'role', 'created_at', 'action_buttons']
    list_filter = ['role', 'created_at']
    search_fields = ['user__username', 'user__email', 'user__first_name', 'user__last_name']
    readonly_fields = ['created_at', 'updated_at']
    list_per_page = 20
    actions = ['assign_admin_role', 'assign_manager_role', 'assign_storage_staff_role']
    
    fieldsets = (
        ('User Information', {
            'fields': ('user', 'role')
        }),
        ('Permissions', {
            'fields': ('permissions',),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_userrole_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def assign_admin_role(self, request, queryset):
        queryset.update(role='admin')
        self.message_user(request, f"{queryset.count()} users assigned admin role.")
    assign_admin_role.short_description = "Assign Admin Role"
    
    def assign_manager_role(self, request, queryset):
        queryset.update(role='manager')
        self.message_user(request, f"{queryset.count()} users assigned manager role.")
    assign_manager_role.short_description = "Assign Manager Role"
    
    def assign_storage_staff_role(self, request, queryset):
        queryset.update(role='storage_staff')
        self.message_user(request, f"{queryset.count()} users assigned storage staff role.")
    assign_storage_staff_role.short_description = "Assign Storage Staff Role"

@admin.register(InventoryItem)
class InventoryItemAdmin(admin.ModelAdmin):
    list_display = ['name', 'sku', 'category', 'client', 'quantity', 'unit_price', 
                   'total_value', 'status_badge', 'location', 'expiry_status', 
                   'created_at', 'action_buttons']
    list_filter = ['status', 'item_type', 'category', 'location', 'client', 'created_at', 'quality_rating']
    search_fields = ['name', 'sku', 'description', 'storage_reference', 'client__username']
    readonly_fields = ['created_at', 'updated_at', 'last_checked', 'total_value']
    #list_editable = ['status', 'quantity', 'unit_price']
    list_per_page = 20
    actions = ['mark_as_active', 'mark_as_reserved', 'mark_as_sold', 'check_low_stock', 
               'check_expiry', 'update_quality_rating']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'sku', 'description', 'category', 'product')
        }),
        ('Inventory Details', {
            'fields': ('quantity', 'unit_price', 'total_value', 'low_stock_threshold', 'reorder_point')
        }),
        ('Status & Type', {
            'fields': ('item_type', 'status', 'quality_rating', 'condition_notes')
        }),
        ('Location & Storage', {
            'fields': ('location', 'storage_reference', 'batch_number')
        }),
        ('Time Information', {
            'fields': ('expiry_date', 'manufactured_date', 'last_checked', 'next_check_date')
        }),
        ('Ownership', {
            'fields': ('client', 'added_by', 'checked_by')
        }),
        ('Dimensions', {
            'fields': ('weight_kg', 'dimensions'),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
    
    def status_badge(self, obj):
        colors = {
            'active': 'success',
            'inactive': 'secondary',
            'reserved': 'warning',
            'sold': 'info',
            'expired': 'danger',
            'damaged': 'dark',
            'returned': 'primary',
        }
        color = colors.get(obj.status, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def expiry_status(self, obj):
        if obj.expiry_date:
            days = obj.days_until_expiry
            if days is not None:
                if days <= 0:
                    return format_html('<span class="badge badge-danger">Expired</span>')
                elif days <= 7:
                    return format_html('<span class="badge badge-warning">{} days</span>', days)
                elif days <= 30:
                    return format_html('<span class="badge badge-info">{} days</span>', days)
                else:
                    return format_html('<span class="badge badge-success">{} days</span>', days)
        return format_html('<span class="badge badge-secondary">No expiry</span>')
    expiry_status.short_description = 'Expiry'
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_inventoryitem_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def mark_as_active(self, request, queryset):
        queryset.update(status='active')
        self.message_user(request, f"{queryset.count()} items marked as active.")
    mark_as_active.short_description = "Mark as active"
    
    def mark_as_reserved(self, request, queryset):
        queryset.update(status='reserved')
        self.message_user(request, f"{queryset.count()} items marked as reserved.")
    mark_as_reserved.short_description = "Mark as reserved"
    
    def mark_as_sold(self, request, queryset):
        queryset.update(status='sold')
        self.message_user(request, f"{queryset.count()} items marked as sold.")
    mark_as_sold.short_description = "Mark as sold"
    
    def check_low_stock(self, request, queryset):
        low_stock_items = []
        for item in queryset:
            if item.is_low_stock:
                low_stock_items.append(item)
        
        if low_stock_items:
            self.message_user(
                request, 
                f"Found {len(low_stock_items)} items with low stock.",
                messages.WARNING
            )
        else:
            self.message_user(request, "No low stock items found.")
    check_low_stock.short_description = "Check low stock"
    
    def check_expiry(self, request, queryset):
        expired_items = []
        near_expiry_items = []
        for item in queryset:
            if item.is_near_expiry:
                if item.days_until_expiry <= 0:
                    expired_items.append(item)
                else:
                    near_expiry_items.append(item)
        
        if expired_items:
            self.message_user(
                request, 
                f"Found {len(expired_items)} expired items.",
                messages.ERROR
            )
        if near_expiry_items:
            self.message_user(
                request, 
                f"Found {len(near_expiry_items)} items near expiry.",
                messages.WARNING
            )
        if not expired_items and not near_expiry_items:
            self.message_user(request, "No expiry issues found.")
    check_expiry.short_description = "Check expiry"
    
    def update_quality_rating(self, request, queryset):
        # This is a placeholder for a more sophisticated quality update
        queryset.update(quality_rating='good')
        self.message_user(request, f"{queryset.count()} items updated with 'Good' quality rating.")
    update_quality_rating.short_description = "Update quality rating"

@admin.register(InventoryHistory)
class InventoryHistoryAdmin(admin.ModelAdmin):
    list_display = ['item', 'action_badge', 'user', 'quantity_change', 'timestamp']
    list_filter = ['action', 'timestamp', 'user']
    search_fields = ['item__name', 'item__sku', 'user__username', 'notes', 'reference_number']
    readonly_fields = ['timestamp']
    list_per_page = 30
    
    def action_badge(self, obj):
        colors = {
            'create': 'success',
            'update': 'info',
            'delete': 'danger',
            'check_in': 'primary',
            'check_out': 'warning',
            'transfer': 'secondary',
            'adjust': 'dark',
            'reserve': 'info',
            'release': 'warning',
            'damage': 'danger',
            'expire': 'dark',
        }
        color = colors.get(obj.action, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_action_display()
        )
    action_badge.short_description = 'Action'
    
    def quantity_change(self, obj):
        if obj.previous_quantity is not None and obj.new_quantity is not None:
            change = obj.new_quantity - obj.previous_quantity
            if change > 0:
                return format_html('<span style="color: green;">+{}</span>', change)
            elif change < 0:
                return format_html('<span style="color: red;">{}</span>', change)
            else:
                return format_html('<span>0</span>')
        return '-'
    quantity_change.short_description = 'Qty Change'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False

@admin.register(Delivery)
class DeliveryAdmin(admin.ModelAdmin):
    list_display = ['delivery_number', 'client_name', 'status_badge', 'estimated_delivery', 
                   'actual_delivery', 'delivery_cost', 'payment_status_badge', 'created_at', 'action_buttons']
    list_filter = ['status', 'payment_status', 'delivery_type', 'created_at', 'estimated_delivery']
    search_fields = ['delivery_number', 'tracking_number', 'client_name', 'client_email', 
                    'client_phone', 'delivery_address']
    readonly_fields = ['created_at', 'updated_at', 'status_changed_at', 'delivery_number', 
                      'tracking_number', 'total_cost']
    #list_editable = ['status', 'payment_status']
    list_per_page = 20
    actions = ['mark_as_processing', 'mark_as_in_transit', 'mark_as_delivered', 'mark_as_cancelled',
               'generate_tracking_numbers', 'update_delivery_status']
    
    fieldsets = (
        ('Delivery Information', {
            'fields': ('delivery_number', 'tracking_number', 'order', 'status')
        }),
        ('Client Information', {
            'fields': ('client', 'client_name', 'client_address', 'client_phone', 'client_email')
        }),
        ('Delivery Details', {
            'fields': ('delivery_address', 'delivery_city', 'delivery_state', 
                      'delivery_country', 'delivery_postal_code')
        }),
        ('Delivery Instructions', {
            'fields': ('special_instructions', 'delivery_type', 'delivery_window_start', 
                      'delivery_window_end')
        }),
        ('Time Tracking', {
            'fields': ('estimated_delivery', 'actual_delivery', 'scheduled_for', 
                      'packed_at', 'shipped_at')
        }),
        ('Proof of Delivery', {
            'fields': ('proof_of_delivery', 'proof_of_delivery_url', 'recipient_name', 
                      'recipient_phone', 'recipient_signature', 'delivery_notes', 
                      'delivery_photos')
        }),
        ('Cost & Payment', {
            'fields': ('delivery_cost', 'delivery_tax', 'total_cost', 'payment_status', 
                      'payment_method', 'insurance_amount')
        }),
        ('Delivery Agent', {
            'fields': ('assigned_to', 'driver_name', 'driver_phone', 'vehicle_number')
        }),
        ('Package Information', {
            'fields': ('package_count', 'total_weight', 'package_dimensions'),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'status_changed_at'),
            'classes': ('collapse',),
        }),
    )
    
    def status_badge(self, obj):
        colors = {
            'pending': 'secondary',
            'processing': 'info',
            'packed': 'primary',
            'in_transit': 'warning',
            'out_for_delivery': 'warning',
            'delivered': 'success',
            'cancelled': 'danger',
            'failed': 'dark',
            'returned': 'dark',
        }
        color = colors.get(obj.status, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def payment_status_badge(self, obj):
        colors = {
            'pending': 'warning',
            'partial': 'info',
            'paid': 'success',
            'overdue': 'danger',
            'refunded': 'secondary',
        }
        color = colors.get(obj.payment_status, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color, obj.get_payment_status_display()
        )
    payment_status_badge.short_description = 'Payment'
    
    def action_buttons(self, obj):
        return format_html(
            '<a href="{}" class="button">View</a>',
            reverse('admin:bika_delivery_change', args=[obj.id])
        )
    action_buttons.short_description = 'Actions'
    
    def mark_as_processing(self, request, queryset):
        queryset.update(status='processing')
        self.message_user(request, f"{queryset.count()} deliveries marked as processing.")
    mark_as_processing.short_description = "Mark as processing"
    
    def mark_as_in_transit(self, request, queryset):
        queryset.update(status='in_transit', shipped_at=timezone.now())
        self.message_user(request, f"{queryset.count()} deliveries marked as in transit.")
    mark_as_in_transit.short_description = "Mark as in transit"
    
    def mark_as_delivered(self, request, queryset):
        queryset.update(status='delivered', actual_delivery=timezone.now())
        self.message_user(request, f"{queryset.count()} deliveries marked as delivered.")
    mark_as_delivered.short_description = "Mark as delivered"
    
    def mark_as_cancelled(self, request, queryset):
        queryset.update(status='cancelled')
        self.message_user(request, f"{queryset.count()} deliveries marked as cancelled.")
    mark_as_cancelled.short_description = "Mark as cancelled"
    
    def generate_tracking_numbers(self, request, queryset):
        import random
        import string
        updated = 0
        for delivery in queryset:
            if not delivery.tracking_number:
                random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                delivery.tracking_number = f"TRK{random_str}"
                delivery.save()
                updated += 1
        self.message_user(request, f"Generated tracking numbers for {updated} deliveries.")
    generate_tracking_numbers.short_description = "Generate tracking numbers"
    
    def update_delivery_status(self, request, queryset):
        # This would typically integrate with a delivery service API
        self.message_user(request, f"Updated status for {queryset.count()} deliveries (simulated).")
    update_delivery_status.short_description = "Update delivery status"

@admin.register(DeliveryItem)
class DeliveryItemAdmin(admin.ModelAdmin):
    list_display = ['delivery', 'item', 'quantity', 'unit_price', 'total_price', 'delivered_quality']
    list_filter = ['delivery__status', 'delivered_quality']
    search_fields = ['delivery__delivery_number', 'item__name', 'item__sku']
    list_editable = ['quantity', 'unit_price', 'delivered_quality']
    list_per_page = 20
    
    def total_price(self, obj):
        return f"${obj.total_price:.2f}"
    total_price.short_description = 'Total'

@admin.register(DeliveryStatusHistory)
class DeliveryStatusHistoryAdmin(admin.ModelAdmin):
    list_display = ['delivery', 'from_status', 'to_status', 'changed_by', 'location', 'timestamp']
    list_filter = ['timestamp', 'changed_by']
    search_fields = ['delivery__delivery_number', 'changed_by__username', 'location', 'notes']
    readonly_fields = ['timestamp']
    list_per_page = 30
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False

# ==================== ADD DASHBOARD TO ADMIN ====================

# Add dashboard to admin URLs
def get_admin_urls():
    def wrap(view):
        def wrapper(*args, **kwargs):
            return admin.site.admin_view(view)(*args, **kwargs)
        return wrapper

    return [
        path('dashboard/', wrap(admin_dashboard), name='admin_dashboard'),
    ]

# Override admin site URLs to include dashboard
original_get_urls = admin.site.get_urls

def custom_get_urls():
    return get_admin_urls() + original_get_urls()

admin.site.get_urls = custom_get_urls
# Customize admin site
admin.site.site_header = "Bika Admin Dashboard"
admin.site.site_title = "Bika Admin"
admin.site.index_title = "Welcome to Bika Administration"

