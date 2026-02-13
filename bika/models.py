# bika/models.py - ALL DJANGO MODELS IN ONE FILE
from datetime import timedelta
from decimal import Decimal
from venv import logger # Added for global use
from django.db import models
from django.urls import reverse
from django.utils import timezone
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator
import random
import string
from django.utils.html import format_html

# ==================== USER MODELS ====================
class CustomUser(AbstractUser):
    """Custom user model with different user types"""
    USER_TYPE_CHOICES = [
        ('admin', 'Administrator'),
        ('manager', 'Manager'),
        ('storage_staff', 'Storage Staff'),
        ('negotiation_team', 'Negotiation Team'),
        ('vendor', 'Vendor'),
        ('customer', 'Customer'),
        ('client', 'Client'),
        ('driver', 'Driver'),
        ('quality_controller', 'Quality Controller'),
        
    ]
    
    ROLE_DESCRIPTIONS = {
        'admin': 'Full system administrator with complete access to all features and settings.',
        'manager': 'Manages operations, staff, and reports. Can oversee multiple departments.',
        'storage_staff': 'Handles inventory management, storage operations, and quality checks.',
        'negotiation_team': 'Responsible for client negotiations, pricing strategies, and contracts.',
        'vendor': 'Sells products on the platform. Manages their own product listings and orders.',
        'customer': 'Regular consumer who browses and purchases products from the platform.',
        'client': 'Business client using specialized services like storage, delivery, and inventory management.',
        'driver': 'Delivery personnel responsible for transporting goods to customers.',
        'quality_controller': 'Ensures product quality standards through inspections and monitoring.',
    }
    
    DEFAULT_PERMISSIONS = {
        'admin': ['all'],
        'manager': ['view_dashboard', 'manage_products', 'manage_orders', 'view_reports', 
                    'manage_staff', 'view_users', 'manage_categories'],
        'storage_staff': ['view_inventory', 'manage_inventory', 'update_stock', 
                          'view_storage_locations', 'update_quality', 'create_alerts'],
        'negotiation_team': ['view_clients', 'manage_contracts', 'set_prices', 
                             'view_negotiations', 'create_proposals'],
        'vendor': ['create_products', 'manage_own_products', 'view_sales', 
                   'view_own_orders', 'update_product_status'],
        'customer': ['browse_products', 'place_orders', 'view_own_orders', 
                     'write_reviews', 'add_to_cart', 'create_wishlist'],
        'client': ['view_own_inventory', 'request_delivery', 'view_storage_status',
                   'manage_own_requests', 'view_invoices', 'download_reports'],
        'driver': ['view_assigned_deliveries', 'update_delivery_status', 
                   'upload_delivery_proof', 'view_delivery_routes'],
        'quality_controller': ['view_quality_readings', 'update_quality_ratings', 
                               'create_quality_alerts', 'inspect_products', 'view_sensor_data'],
    }
    
    user_type = models.CharField(
        max_length=20, 
        choices=USER_TYPE_CHOICES, 
        default='customer',
        help_text="Primary user classification that determines base permissions"
    )
    
    phone = models.CharField(max_length=20, blank=True, help_text="Primary contact number")
    company = models.CharField(max_length=100, blank=True, help_text="Company or organization name")
    address = models.TextField(blank=True, help_text="Complete physical address")
    profile_picture = models.ImageField(upload_to='profiles/%Y/%m/', blank=True, null=True)
    
    # Verification fields
    email_verified = models.BooleanField(default=False)
    phone_verified = models.BooleanField(default=False)
    identity_verified = models.BooleanField(default=False, help_text="Government ID verification")
    
    # Additional fields for vendors
    business_name = models.CharField(max_length=200, blank=True, help_text="Registered business name")
    business_description = models.TextField(blank=True)
    business_logo = models.ImageField(upload_to='business_logos/%Y/%m/', blank=True, null=True)
    business_verified = models.BooleanField(default=False)
    business_registration_number = models.CharField(max_length=100, blank=True)
    tax_id = models.CharField(max_length=100, blank=True, verbose_name="Tax ID/VAT Number")
    
    # Additional fields for drivers
    driver_license_number = models.CharField(max_length=50, blank=True)
    vehicle_registration = models.CharField(max_length=50, blank=True)
    license_expiry_date = models.DateField(null=True, blank=True)
    
    # Additional fields for quality controllers
    certification_number = models.CharField(max_length=100, blank=True)
    certification_authority = models.CharField(max_length=200, blank=True)
    
    # Status fields
    is_approved = models.BooleanField(default=True, help_text="Whether user is approved to use the system")
    approval_date = models.DateTimeField(null=True, blank=True)
    approved_by = models.ForeignKey(
        'self', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='approved_users'
    )
    
    # Communication preferences
    receive_marketing_emails = models.BooleanField(default=True)
    receive_sms_notifications = models.BooleanField(default=True)
    receive_push_notifications = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_active = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-date_joined']
        verbose_name = "User"
        verbose_name_plural = "Users"
        indexes = [
            models.Index(fields=['user_type', 'is_active']),
            models.Index(fields=['email_verified', 'phone_verified']),
            models.Index(fields=['business_verified']),
        ]
    
    def __str__(self):
        display_name = self.business_name if self.business_name and self.is_vendor() else self.get_full_name()
        if not display_name:
            display_name = self.username
        return f"{display_name} ({self.get_user_type_display()})"
    
    def save(self, *args, **kwargs):
        # Set approval date if user is being approved
        if self.pk:
            original = CustomUser.objects.get(pk=self.pk)
            if not original.is_approved and self.is_approved and not self.approval_date:
                self.approval_date = timezone.now()
        
        # Generate username if not provided (for certain user types)
        if not self.username and self.email:
            base_username = self.email.split('@')[0]
            username = base_username
            counter = 1
            while CustomUser.objects.filter(username=username).exists():
                username = f"{base_username}{counter}"
                counter += 1
            self.username = username
        
        super().save(*args, **kwargs)
    
    # Role checking methods
    def is_vendor(self):
        return self.user_type == 'vendor'
    
    def is_customer(self):
        return self.user_type == 'customer'
    
    def is_client(self):
        return self.user_type == 'client'
    
    def is_staff_member(self):
        """Check if user is any type of staff (not customer/vendor/client)"""
        staff_types = ['admin', 'manager', 'storage_staff', 'negotiation_team', 
                      'driver', 'quality_controller']
        return self.user_type in staff_types
    
    def is_admin_or_manager(self):
        """Check if user has admin or manager privileges"""
        return self.user_type in ['admin', 'manager']
    
    def is_quality_staff(self):
        """Check if user is involved in quality control"""
        return self.user_type in ['quality_controller', 'storage_staff']
    
    # Permission methods
    def get_role_description(self):
        """Get human-readable role description"""
        return self.ROLE_DESCRIPTIONS.get(self.user_type, 'No description available')
    
    def get_default_permissions(self):
        """Get default permissions for this user type"""
        return self.DEFAULT_PERMISSIONS.get(self.user_type, [])
    
    def has_permission(self, permission_code):
        """
        Check if user has a specific permission based on their role.
        This works in conjunction with UserRole model.
        """
        try:
            # First check UserRole permissions
            if hasattr(self, 'user_role'):
                return self.user_role.has_permission(permission_code)
            
            # Fallback to default permissions for this user type
            default_perms = self.get_default_permissions()
            if 'all' in default_perms:
                return True
            return permission_code in default_perms
        except Exception:
            return False
    
    # Business logic methods
    def can_list_products(self):
        """Check if user can list products for sale"""
        return self.user_type in ['vendor', 'admin', 'manager']
    
    def can_manage_inventory(self):
        """Check if user can manage inventory"""
        return self.user_type in ['storage_staff', 'admin', 'manager', 'client']
    
    def can_view_dashboard(self):
        """Check if user can view admin dashboard"""
        return self.user_type in ['admin', 'manager', 'storage_staff', 'negotiation_team']
    
    def can_manage_deliveries(self):
        """Check if user can manage deliveries"""
        return self.user_type in ['driver', 'admin', 'manager', 'storage_staff']
    
    def can_inspect_quality(self):
        """Check if user can perform quality inspections"""
        return self.user_type in ['quality_controller', 'storage_staff', 'admin', 'manager']
    
    # Verification methods
    def is_fully_verified(self):
        """Check if user has completed all verification steps"""
        if self.is_vendor():
            return all([
                self.email_verified,
                self.phone_verified,
                self.business_verified,
                self.identity_verified
            ])
        elif self.user_type in ['driver', 'quality_controller']:
            return all([
                self.email_verified,
                self.phone_verified,
                self.identity_verified
            ])
        else:
            return self.email_verified and self.phone_verified
    
    def get_verification_status(self):
        """Get verification status as percentage"""
        required_verifications = 2  # email + phone (minimum)
        
        if self.is_vendor():
            required_verifications = 4  # email, phone, business, identity
        elif self.user_type in ['driver', 'quality_controller']:
            required_verifications = 3  # email, phone, identity
        
        verified_count = sum([
            self.email_verified,
            self.phone_verified,
            self.business_verified if hasattr(self, 'business_verified') else False,
            self.identity_verified
        ])
        
        return (verified_count / required_verifications) * 100
    
    # Display methods
    def get_display_name(self):
        """Get the best display name for the user"""
        if self.business_name and self.is_vendor():
            return self.business_name
        elif self.get_full_name():
            return self.get_full_name()
        else:
            return self.username
    
    def get_role_badge(self):
        """Get HTML badge for user role"""
        role_colors = {
            'admin': 'danger',
            'manager': 'warning',
            'storage_staff': 'info',
            'negotiation_team': 'primary',
            'vendor': 'success',
            'customer': 'secondary',
            'client': 'dark',
            'driver': 'light',
            'quality_controller': 'info',
        }
        
        color = role_colors.get(self.user_type, 'secondary')
        return format_html(
            '<span class="badge badge-{}">{}</span>',
            color,
            self.get_user_type_display()
        )
    
    # Activity methods
    def update_last_active(self):
        """Update last active timestamp"""
        self.last_active = timezone.now()
        self.save(update_fields=['last_active'])
    
    def get_activity_status(self):
        """Get user activity status"""
        if not self.last_active:
            return "Never active"
        
        delta = timezone.now() - self.last_active
        if delta.days < 1:
            return "Active today"
        elif delta.days < 7:
            return f"Active {delta.days} days ago"
        else:
            return f"Inactive for {delta.days} days"
    
    # Model URLs
    def get_absolute_url(self):
        return reverse('bika:user_profile', kwargs={'username': self.username})
    
    def get_admin_url(self):
        return reverse('admin:bika_customuser_change', args=[self.id])


# ==================== ADDRESS MODEL (MOVED OUTSIDE CustomUser) ====================
class Address(models.Model):
    """User Address Book"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='addresses')
    title = models.CharField(max_length=100, blank=True, help_text="e.g., Home, Work, Aunt's House")
    full_name = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=20)
    street_address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100, blank=True)
    postal_code = models.CharField(max_length=20, blank=True)
    country = models.CharField(max_length=100, default="Tanzania")
    is_default_shipping = models.BooleanField(default=False)
    is_default_billing = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "Addresses"
        ordering = ['-created_at']
        unique_together = ('user', 'title')
    
    def __str__(self):
        return f"{self.full_name} - {self.street_address}, {self.city}"
    
    def save(self, *args, **kwargs):
        # Ensure only one default shipping address per user
        if self.is_default_shipping:
            Address.objects.filter(user=self.user, is_default_shipping=True).exclude(pk=self.pk).update(is_default_shipping=False)
        # Ensure only one default billing address per user
        if self.is_default_billing:
            Address.objects.filter(user=self.user, is_default_billing=True).exclude(pk=self.pk).update(is_default_billing=False)
        super().save(*args, **kwargs)


# ==================== CORE MODELS ====================
class ProductCategory(models.Model):
    """Product category model"""
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='categories/', blank=True, null=True)
    display_order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='subcategories')
    
    class Meta:
        verbose_name_plural = "Product Categories"
        ordering = ['display_order', 'name']
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('bika:products_by_category', kwargs={'category_slug': self.slug})


class Product(models.Model):
    """Main product model"""
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('active', 'Active'),
        ('out_of_stock', 'Out of Stock'),
        ('discontinued', 'Discontinued'),
    ]
    
    CONDITION_CHOICES = [
        ('new', 'New'),
        ('refurbished', 'Refurbished'),
        ('used_like_new', 'Used - Like New'),
        ('used_good', 'Used - Good'),
        ('used_fair', 'Used - Fair'),
    ]
    
    # Basic Information
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    sku = models.CharField(max_length=100, unique=True, verbose_name="SKU")
    barcode = models.CharField(max_length=100, blank=True, unique=True, null=True)
    description = models.TextField()
    short_description = models.TextField(max_length=300, blank=True)
    image = models.ImageField(upload_to='products/', blank=True, null=True) # Main product image

    # Categorization
    category = models.ForeignKey(ProductCategory, on_delete=models.CASCADE, related_name='products')
    tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated tags")
    
    # Pricing
    price = models.DecimalField(max_digits=10, decimal_places=2)
    compare_price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True, 
                                      verbose_name="Compare at Price")
    cost_price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True,
                                   verbose_name="Cost Price")
    tax_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0.0,
                                 verbose_name="Tax Rate (%)")
    
    # Inventory
    stock_quantity = models.IntegerField(default=0)
    low_stock_threshold = models.IntegerField(default=5, verbose_name="Low Stock Alert")
    track_inventory = models.BooleanField(default=True)
    allow_backorders = models.BooleanField(default=False)
    
    # Product Details
    brand = models.CharField(max_length=100, blank=True)
    model = models.CharField(max_length=100, blank=True)
    weight = models.DecimalField(max_digits=8, decimal_places=2, blank=True, null=True, help_text="Weight in kg")
    dimensions = models.CharField(max_length=100, blank=True, help_text="L x W x H in cm")
    color = models.CharField(max_length=50, blank=True)
    size = models.CharField(max_length=50, blank=True)
    material = models.CharField(max_length=100, blank=True)
    
    # Status & Visibility
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    condition = models.CharField(max_length=20, choices=CONDITION_CHOICES, default='new')
    is_featured = models.BooleanField(default=False)
    is_digital = models.BooleanField(default=False, verbose_name="Digital Product")

    PRODUCT_TYPE_CHOICES = [
        ('public', 'Public E-commerce Product'),
        ('private', 'Private Client Stock Template'),
    ]
    product_type = models.CharField(
        max_length=20,
        choices=PRODUCT_TYPE_CHOICES,
        default='public',
        help_text="Determines if the product is for public sale or a template for private client stock."
    )
    
    # Client-specific fields
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='owned_products',
                             limit_choices_to={'user_type': 'client'}, null=True, blank=True)
    storage_charges = models.DecimalField(max_digits=10, decimal_places=2, default=0.00,
                                         help_text="Charges to the client for storing this product.")
    client_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00,
                                      help_text="Price charged to the client for this product.")
    is_approved = models.BooleanField(default=False, help_text="Whether this product has been approved by an admin.")
    is_available = models.BooleanField(default=True, help_text="Whether this product is currently available for sale.")
    
    # Vendor Information
    vendor = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='listed_products', limit_choices_to={'user_type': 'vendor'})
    
    # Supplier Information (NEW)
    default_supplier = models.ForeignKey('Supplier', on_delete=models.SET_NULL, null=True, blank=True, related_name='supplied_products')
    
    # SEO
    meta_title = models.CharField(max_length=200, blank=True)
    meta_description = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(blank=True, null=True)
    
    views_count = models.PositiveIntegerField(default=0, verbose_name="View Count")
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.sku}"
    
    def get_absolute_url(self):
        return reverse('bika:product_detail', kwargs={'slug': self.slug})
    
    def save(self, *args, **kwargs):
        if self.status == 'active' and not self.published_at:
            self.published_at = timezone.now()
        super().save(*args, **kwargs)
    
    @property
    def is_in_stock(self):
        if not self.track_inventory:
            return True
        return self.stock_quantity > 0
    
    @property
    def is_low_stock(self):
        if not self.track_inventory:
            return False
        return 0 < self.stock_quantity <= self.low_stock_threshold
    
    @property
    def discount_percentage(self):
        if self.compare_price and self.compare_price > self.price:
            return round(((self.compare_price - self.price) / self.compare_price) * 100, 1)
        return 0
    
    @property
    def final_price(self):
        return self.price
    
    def get_related_products(self, limit=4):
        return Product.objects.filter(
            category=self.category,
            status='active'
        ).exclude(id=self.id)[:limit]


class ProductImage(models.Model):
    """Product images model"""
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='products/')
    alt_text = models.CharField(max_length=200, blank=True)
    display_order = models.IntegerField(default=0)
    is_primary = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['display_order', 'id']
    
    def __str__(self):
        return f"Image for {self.product.name}"
    
    def save(self, *args, **kwargs):
        if self.is_primary:
            # Ensure only one primary image per product
            ProductImage.objects.filter(product=self.product, is_primary=True).update(is_primary=False)
        super().save(*args, **kwargs)


class ProductReview(models.Model):
    """Product reviews model"""
    RATING_CHOICES = [
        (1, '1 Star'),
        (2, '2 Stars'),
        (3, '3 Stars'),
        (4, '4 Stars'),
        (5, '5 Stars'),
    ]
    
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=RATING_CHOICES)
    title = models.CharField(max_length=200)
    comment = models.TextField()
    is_verified_purchase = models.BooleanField(default=False)
    is_approved = models.BooleanField(default=False)
    helpful_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['product', 'user']
    
    def __str__(self):
        return f"Review by {self.user.username} for {self.product.name}"


# ==================== E-COMMERCE MODELS ====================
class Wishlist(models.Model):
    """User wishlist model"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'product']
        ordering = ['-added_at']
    
    def __str__(self):
        return f"{self.user.username}'s wishlist - {self.product.name}"


class Cart(models.Model):
    """Shopping cart model"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    added_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'product']
        ordering = ['-added_at']
    
    def __str__(self):
        return f"{self.user.username}'s cart - {self.product.name}"
    
    @property
    def total_price(self):
        from decimal import Decimal
        # Ensure both are Decimal
        price = Decimal(str(self.product.price))
        quantity = Decimal(str(self.quantity))
        return price * quantity
    

class Order(models.Model):
    """Order model"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    order_number = models.CharField(max_length=20, unique=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    shipping_address = models.TextField()
    billing_address = models.TextField()
    shipping_method = models.ForeignKey('ShippingMethod', on_delete=models.SET_NULL, null=True, blank=True, related_name='orders')
    shipping_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    
    coupon = models.ForeignKey('Coupon', on_delete=models.SET_NULL, null=True, blank=True)
    discount_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Order #{self.order_number} - {self.user.username}"
    
    def save(self, *args, **kwargs):
        if not self.order_number:
            random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            self.order_number = f"ORD{timezone.now().strftime('%Y%m%d')}{random_str}"
        super().save(*args, **kwargs)


class OrderItem(models.Model):
    """Order items model"""
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def __str__(self):
        return f"{self.product.name} - {self.order.order_number}"
    
    @property
    def total_price(self):
        return self.price * self.quantity


# ==================== FRUIT MONITORING MODELS ====================
class FruitType(models.Model):
    """Different types of fruits"""
    name = models.CharField(max_length=100, unique=True)
    scientific_name = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='fruits/', blank=True, null=True)
    description = models.TextField(blank=True)
    
    # Optimal storage conditions for each fruit type
    optimal_temp_min = models.DecimalField(max_digits=5, decimal_places=2, default=2.0)
    optimal_temp_max = models.DecimalField(max_digits=5, decimal_places=2, default=8.0)
    optimal_humidity_min = models.DecimalField(max_digits=5, decimal_places=2, default=85.0)
    optimal_humidity_max = models.DecimalField(max_digits=5, decimal_places=2, default=95.0)
    optimal_light_max = models.IntegerField(default=100)  # Maximum light (lux)
    optimal_co2_max = models.IntegerField(default=400)    # Maximum CO₂ (ppm)
    
    # Shelf life information
    shelf_life_days = models.IntegerField(default=7)
    ethylene_sensitive = models.BooleanField(default=False)
    chilling_sensitive = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name


class FruitBatch(models.Model):
    """Batch of fruits for monitoring"""
    BATCH_STATUS = [
        ('pending', 'Pending'),
        ('active', 'Active Monitoring'),
        ('completed', 'Completed'),
        ('discarded', 'Discarded'),
    ]
    
    batch_number = models.CharField(max_length=50, unique=True)
    fruit_type = models.ForeignKey(FruitType, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    quantity = models.IntegerField(default=0)
    arrival_date = models.DateTimeField(default=timezone.now)
    expected_expiry = models.DateTimeField()
    supplier = models.CharField(max_length=200, blank=True)
    storage_location = models.ForeignKey('StorageLocation', on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=BATCH_STATUS, default='pending')
    
    # Initial quality assessment
    initial_quality = models.CharField(max_length=20, choices=[
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('fair', 'Fair'),
        ('poor', 'Poor'),
    ], default='good')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "Fruit Batches"
    
    def __str__(self):
        return f"{self.batch_number} - {self.fruit_type.name}"
    
    @property
    def days_remaining(self):
        """Calculate days until expected expiry"""
        if self.expected_expiry:
            remaining = (self.expected_expiry - timezone.now()).days
            return max(remaining, 0)
        return 0


# Update Product model to include fruit relationships (add these to existing Product model)
# Note: We've already added these fields to the Product model above

class FruitQualityReading(models.Model):
    """Sensor readings and quality predictions for fruit batches"""
    QUALITY_CLASSES = [
        ('Fresh', 'Fresh'),
        ('Good', 'Good'),
        ('Fair', 'Fair'),
        ('Poor', 'Poor'),
        ('Rotten', 'Rotten'),
    ]
    
    fruit_batch = models.ForeignKey(FruitBatch, on_delete=models.CASCADE, related_name='quality_readings')
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Sensor readings
    temperature = models.DecimalField(max_digits=5, decimal_places=2)
    humidity = models.DecimalField(max_digits=5, decimal_places=2)
    light_intensity = models.DecimalField(max_digits=10, decimal_places=2, help_text="Light in lux")
    co2_level = models.IntegerField()
    
    # Quality assessment
    actual_class = models.CharField(max_length=20, choices=QUALITY_CLASSES, blank=True)
    predicted_class = models.CharField(max_length=20, choices=QUALITY_CLASSES)
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    
    # Additional metrics
    ethylene_level = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, help_text="Ethylene in ppm")
    weight_loss = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, help_text="Weight loss percentage")
    firmness = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, help_text="Firmness in N")
    
    # AI model info
    model_used = models.CharField(max_length=50, blank=True)
    model_version = models.CharField(max_length=20, blank=True)
    
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['fruit_batch', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.fruit_batch.batch_number} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


# ==================== STORAGE & SENSOR MODELS ====================
class StorageLocation(models.Model):
    """Storage location for products"""
    name = models.CharField(max_length=200)
    address = models.TextField()
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    capacity = models.IntegerField(default=0)
    current_occupancy = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    @property
    def available_capacity(self):
        return self.capacity - self.current_occupancy
    
    # ADD THIS PROPERTY METHOD:
    @property
    def occupancy_percentage(self):
        """Calculate occupancy as a percentage"""
        if self.capacity > 0:
            return (self.current_occupancy / self.capacity) * 100
        return 0


class RealTimeSensorData(models.Model):
    """Real-time sensor data model"""
    SENSOR_TYPES = [
        ('temperature', 'Temperature'),
        ('humidity', 'Humidity'),
        ('light', 'Light Intensity'),
        ('co2', 'CO₂ Level'),
        ('ethylene', 'Ethylene'),
        ('weight', 'Weight'),
        ('firmness', 'Firmness'),
        ('color', 'Color'),
        ('vibration', 'Vibration'),
        ('pressure', 'Pressure'),
    ]
    
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=True, blank=True)
    fruit_batch = models.ForeignKey(FruitBatch, on_delete=models.CASCADE, null=True, blank=True)
    sensor_type = models.CharField(max_length=50, choices=SENSOR_TYPES)
    value = models.FloatField()
    unit = models.CharField(max_length=20)
    location = models.ForeignKey(StorageLocation, on_delete=models.CASCADE, null=True, blank=True)
    recorded_at = models.DateTimeField(auto_now_add=True)
    
    # Quality prediction
    predicted_class = models.CharField(max_length=20, blank=True)
    condition_confidence = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    
    class Meta:
        ordering = ['-recorded_at']
        indexes = [
            models.Index(fields=['product', 'sensor_type', 'recorded_at']),
        ]
    
    def __str__(self):
        return f"{self.sensor_type} - {self.value}{self.unit}"


# ==================== AI & DATASET MODELS ====================
class ProductDataset(models.Model):
    """Product datasets for AI training"""
    DATASET_TYPES = [
        ('anomaly_detection', 'Anomaly Detection'),
        ('sales_forecast', 'Sales Forecasting'),
        ('inventory_optimization', 'Inventory Optimization'),
        ('quality_control', 'Quality Control'),
    ]
    
    name = models.CharField(max_length=200)
    dataset_type = models.CharField(max_length=50, choices=DATASET_TYPES)
    description = models.TextField()
    data_file = models.FileField(upload_to='datasets/')
    columns = models.JSONField(default=dict)  # Store column metadata
    row_count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.get_dataset_type_display()})"


class TrainedModel(models.Model):
    """Trained AI models"""
    MODEL_TYPES = [
        ('anomaly_detection', 'Anomaly Detection'),
        ('sales_forecast', 'Sales Forecasting'),
        ('stock_prediction', 'Stock Prediction'),
        ('fruit_quality', 'Fruit Quality Prediction'),
    ]
    
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    dataset = models.ForeignKey(ProductDataset, on_delete=models.CASCADE)
    model_file = models.FileField(upload_to='trained_models/')
    accuracy = models.FloatField(null=True, blank=True)
    training_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    feature_columns = models.JSONField(default=list)
    
    def __str__(self):
        return f"{self.name} - {self.get_model_type_display()}"


# ==================== ALERT & NOTIFICATION MODELS ====================
class ProductAlert(models.Model):
    """Product quality and stock alerts"""
    ALERT_TYPES = [
        ('stock_low', 'Low Stock'),
        ('expiry_near', 'Near Expiry'),
        ('quality_issue', 'Quality Issue'),
        ('temperature_anomaly', 'Temperature Anomaly'),
        ('humidity_issue', 'Humidity Issue'),
        ('ai_anomaly', 'AI Detected Anomaly'),
        ('predicted_low_stock', 'Predicted Low Stock'), # New AI Alert Type
        ('predicted_expiry', 'Predicted Expiry'),       # New AI Alert Type
        ('predicted_quality_issue', 'Predicted Quality Issue'), # New AI Alert Type
        ('predicted_high_demand', 'Predicted High Demand'), # New AI Alert Type
        ('movement_anomaly', 'Inventory Movement Anomaly'), # NEW
        ('transfer_delay', 'Inventory Transfer Delay'),     # NEW
        ('delivery_delay', 'Delivery Delay'),               # NEW
        ('driver_issue', 'Driver Issue Alert'),             # NEW
    ]
    
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=True, blank=True, help_text="The general product definition (optional if inventory_item is set).")
    inventory_item = models.ForeignKey('InventoryItem', on_delete=models.CASCADE, null=True, blank=True, related_name='alerts', help_text="The specific inventory item instance (if alert is item-specific).")
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPES)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    message = models.TextField()
    detected_by = models.CharField(max_length=50)  # ai_system, sensor_system, manual
    is_resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Link to Delivery (NEW)
    delivery = models.ForeignKey('Delivery', on_delete=models.SET_NULL, null=True, blank=True, related_name='alerts')
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['product']),
            models.Index(fields=['inventory_item']),
            models.Index(fields=['alert_type']),
            models.Index(fields=['severity']),
            models.Index(fields=['is_resolved']),
        ]
    
    def __str__(self):
        if self.inventory_item:
            return f"{self.get_alert_type_display()} - {self.inventory_item.name}"
        elif self.product:
            return f"{self.get_alert_type_display()} - {self.product.name}"
        return f"{self.get_alert_type_display()} - ID: {self.pk}"

class Notification(models.Model):
    """User notifications"""
    NOTIFICATION_TYPES = [
        ('product_alert', 'Product Alert'),
        ('order_update', 'Order Update'),
        ('system_alert', 'System Alert'),
        ('urgent_alert', 'Urgent Alert'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=50, choices=NOTIFICATION_TYPES)
    is_read = models.BooleanField(default=False)
    related_object_type = models.CharField(max_length=100, blank=True)
    related_object_id = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"


class NotificationSettings(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='notification_settings')
    email_notifications = models.BooleanField(default=True)
    push_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)
    
    # Specific notification types
    order_updates = models.BooleanField(default=True)
    delivery_updates = models.BooleanField(default=True)
    inventory_alerts = models.BooleanField(default=True)
    quality_alerts = models.BooleanField(default=True)
    system_alerts = models.BooleanField(default=True)
    promotions = models.BooleanField(default=False)

    class Meta:
        verbose_name_plural = "Notification Settings"

    def __str__(self):
        return f"Notification Settings for {self.user.username}"


class TwoFactorSettings(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='two_factor_settings')
    is_enabled = models.BooleanField(default=False)
    method = models.CharField(max_length=50, choices=[
        ('app', 'Authenticator App'),
        ('sms', 'SMS'),
        ('email', 'Email'),
    ], default='app')
    phone_number = models.CharField(max_length=20, blank=True) # For SMS method
    # Add fields for authenticator app secret key, etc., as needed for a full implementation

    class Meta:
        verbose_name_plural = "Two-Factor Settings"

    def __str__(self):
        return f"2FA Settings for {self.user.username} (Enabled: {self.is_enabled})"

# ==================== MARKETING MODELS ====================
class MarketingCampaign(models.Model):
    """Model for defining and tracking marketing campaigns."""
    CAMPAIGN_TYPES = [
        ('email', 'Email Campaign'),
        ('sms', 'SMS Campaign'),
        ('in_app', 'In-App Notification'),
        ('push', 'Push Notification'),
        ('banner', 'Website Banner'),
    ]

    TARGET_SEGMENTS = [
        ('all_users', 'All Users'),
        ('customers', 'All Customers'),
        ('vendors', 'All Vendors'),
        ('new_users', 'New Users (last 30 days)'),
        ('inactive_users', 'Inactive Users (last 90 days)'),
        ('high_value_customers', 'High-Value Customers'),
        ('cart_abandoners', 'Cart Abandoners'),
        ('category_interest', 'Users Interested in Specific Categories'), # Requires additional logic
    ]

    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('scheduled', 'Scheduled'),
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]

    name = models.CharField(max_length=200, help_text="Name of the marketing campaign")
    description = models.TextField(blank=True, help_text="Detailed description of the campaign objectives and content")
    
    campaign_type = models.CharField(max_length=20, choices=CAMPAIGN_TYPES, default='email')
    target_segment = models.CharField(max_length=50, choices=TARGET_SEGMENTS, default='all_users')
    
    # Content fields (can be JSON for richer content or HTML for emails)
    subject = models.CharField(max_length=255, blank=True, help_text="Subject line for email/push notifications")
    content = models.TextField(help_text="Main content of the campaign message (e.g., email body, in-app text)")
    
    # Scheduling
    start_date = models.DateTimeField(default=timezone.now)
    end_date = models.DateTimeField(null=True, blank=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    
    # Tracking metrics
    sent_count = models.PositiveIntegerField(default=0)
    opened_count = models.PositiveIntegerField(default=0) # For email/push
    clicked_count = models.PositiveIntegerField(default=0) # For email/push/banner
    conversion_count = models.PositiveIntegerField(default=0) # e.g., orders placed from campaign
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_campaigns')

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Marketing Campaign"
        verbose_name_plural = "Marketing Campaigns"

    def __str__(self):
        return self.name

    def is_active_campaign(self):
        """Check if the campaign is currently active."""
        now = timezone.now()
        return self.status == 'active' and self.start_date <= now and (self.end_date is None or now <= self.end_date)

    def get_target_users(self):
        """
        Returns a QuerySet of CustomUser objects that match the target segment.
        This method will need to be expanded with more sophisticated segmentation logic.
        """
        users = CustomUser.objects.filter(is_active=True)
        now = timezone.now()

        if self.target_segment == 'customers':
            users = users.filter(user_type='customer')
        elif self.target_segment == 'vendors':
            users = users.filter(user_type='vendor')
        elif self.target_segment == 'new_users':
            users = users.filter(date_joined__gte=now - timedelta(days=30))
        elif self.target_segment == 'inactive_users':
            users = users.filter(last_login__lt=now - timedelta(days=90))
        # Add more sophisticated logic for 'high_value_customers', 'cart_abandoners', 'category_interest' later
        # For 'all_users' or no specific filter, it returns all active users.
        return users
# ==================== PAYMENT MODELS ====================
class Payment(models.Model):
    """Payment model"""
    PAYMENT_METHODS = [
        # Mobile Money - Tanzania
        ('mpesa', 'M-Pesa (TZ)'),
        ('tigo_tz', 'Tigo Pesa (TZ)'),
        ('airtel_tz', 'Airtel Money (TZ)'),
        ('halotel_tz', 'Halotel (TZ)'),
        
        # Mobile Money - Rwanda
        ('mtn_rw', 'MTN Mobile Money (RW)'),
        ('airtel_rw', 'Airtel Money (RW)'),
        
        # Mobile Money - Uganda
        ('mtn_ug', 'MTN Mobile Money (UG)'),
        ('airtel_ug', 'Airtel Money (UG)'),
        
        # Mobile Money - Kenya
        ('mpesa_ke', 'M-Pesa (KE)'),
        
        # Cards & International
        ('visa', 'Visa Card'),
        ('mastercard', 'MasterCard'),
        ('amex', 'American Express'),
        ('paypal', 'PayPal'),
        ('bank_transfer', 'Bank Transfer'),
    ]
    
    PAYMENT_STATUS = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
        ('refunded', 'Refunded'),
    ]
    
    CURRENCIES = [
        ('TZS', 'Tanzanian Shilling'),
        ('RWF', 'Rwandan Franc'),
        ('UGX', 'Ugandan Shilling'),
        ('KES', 'Kenyan Shilling'),
        ('USD', 'US Dollar'),
        ('EUR', 'Euro'),
    ]
    
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='payments')
    payment_method = models.CharField(max_length=20, choices=PAYMENT_METHODS)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, choices=CURRENCIES, default='TZS')
    status = models.CharField(max_length=20, choices=PAYMENT_STATUS, default='pending')
    transaction_id = models.CharField(max_length=100, blank=True, unique=True)
    
    # Mobile Money fields
    mobile_money_phone = models.CharField(max_length=20, blank=True)
    mobile_money_provider = models.CharField(max_length=50, blank=True)
    mobile_money_transaction_id = models.CharField(max_length=100, blank=True)
    
    # Card fields
    card_last4 = models.CharField(max_length=4, blank=True)
    card_brand = models.CharField(max_length=20, blank=True)
    card_country = models.CharField(max_length=2, blank=True)
    
    # International payment fields
    payer_email = models.EmailField(blank=True)
    payer_country = models.CharField(max_length=2, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['transaction_id']),
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"Payment #{self.id} - {self.amount} {self.currency}"
    
    def is_successful(self):
        return self.status == 'completed'


class PaymentGatewaySettings(models.Model):
    """Enhanced payment gateway configuration"""
    GATEWAY_CHOICES = [
        # Tanzania
        ('mpesa_tz', 'M-Pesa Tanzania'),
        ('tigo_tz', 'Tigo Pesa Tanzania'),
        ('airtel_tz', 'Airtel Money Tanzania'),
        ('halotel_tz', 'Halotel Tanzania'),
        
        # Rwanda
        ('mtn_rw', 'MTN Rwanda'),
        ('airtel_rw', 'Airtel Rwanda'),
        
        # Uganda
        ('mtn_ug', 'MTN Uganda'),
        ('airtel_ug', 'Airtel Uganda'),
        
        # Kenya
        ('mpesa_ke', 'M-Pesa Kenya'),
        
        # International
        ('stripe', 'Stripe'),
        ('paypal', 'PayPal'),
    ]
    
    gateway = models.CharField(max_length=20, choices=GATEWAY_CHOICES, unique=True)
    is_active = models.BooleanField(default=False)
    display_name = models.CharField(max_length=100, blank=True)
    supported_countries = models.JSONField(default=list)
    supported_currencies = models.JSONField(default=list)
    
    # API Credentials
    api_key = models.CharField(max_length=255, blank=True)
    api_secret = models.CharField(max_length=255, blank=True)
    merchant_id = models.CharField(max_length=100, blank=True)
    webhook_secret = models.CharField(max_length=255, blank=True)
    api_user = models.CharField(max_length=100, blank=True, help_text="Specific API User ID for some gateways (e.g., MTN MoMo)")
    api_password = models.CharField(max_length=255, blank=True, help_text="Specific API Password for some gateways (e.g., MTN MoMo API User's API Key)")
    
    # Configuration
    base_url = models.URLField(blank=True)
    callback_url = models.URLField(blank=True)
    environment = models.CharField(max_length=10, default='sandbox', choices=[('sandbox', 'Sandbox'), ('live', 'Live')])
    
    # Fees
    transaction_fee_percent = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    transaction_fee_fixed = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.get_gateway_display()} Settings"


class CurrencyExchangeRate(models.Model):
    """Currency exchange rates"""
    base_currency = models.CharField(max_length=3)
    target_currency = models.CharField(max_length=3)
    exchange_rate = models.DecimalField(max_digits=10, decimal_places=6)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['base_currency', 'target_currency']
    
    def __str__(self):
        return f"{self.base_currency}/{self.target_currency}: {self.exchange_rate}"


# ==================== SITE CONTENT MODELS ====================
class SiteInfo(models.Model):
    """Store site-wide information"""
    name = models.CharField(max_length=200, default="Bika")
    tagline = models.CharField(max_length=300, blank=True)
    description = models.TextField(blank=True)
    email = models.EmailField(default="contact@bika.com")
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    logo = models.ImageField(upload_to='site/logo/', blank=True, null=True)
    favicon = models.ImageField(upload_to='site/favicon/', blank=True, null=True)
    
    # Social Media
    facebook_url = models.URLField(blank=True)
    twitter_url = models.URLField(blank=True)
    instagram_url = models.URLField(blank=True)
    linkedin_url = models.URLField(blank=True)
    
    # SEO
    meta_title = models.CharField(max_length=200, blank=True)
    meta_description = models.TextField(blank=True)
    
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Site Information"
        verbose_name_plural = "Site Information"
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        if not self.pk and SiteInfo.objects.exists():
            existing = SiteInfo.objects.first()
            existing.name = self.name
            existing.tagline = self.tagline
            existing.description = self.description
            existing.email = self.email
            existing.phone = self.phone
            existing.address = self.address
            if self.logo:
                existing.logo = self.logo
            if self.favicon:
                existing.favicon = self.favicon
            existing.facebook_url = self.facebook_url
            existing.twitter_url = self.twitter_url
            existing.instagram_url = self.instagram_url
            existing.linkedin_url = self.linkedin_url
            existing.meta_title = self.meta_title
            existing.meta_description = self.meta_description
            existing.save()
            return
        super().save(*args, **kwargs)


class Service(models.Model):
    """Services offered by Bika"""
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField()
    icon = models.CharField(max_length=100, help_text="Font Awesome icon class")
    image = models.ImageField(upload_to='services/', blank=True, null=True)
    display_order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['display_order', 'name']
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('bika:service_detail', kwargs={'slug': self.slug})


class Testimonial(models.Model):
    """Customer testimonials"""
    name = models.CharField(max_length=200)
    position = models.CharField(max_length=200, blank=True)
    company = models.CharField(max_length=200, blank=True)
    content = models.TextField()
    image = models.ImageField(upload_to='testimonials/', blank=True, null=True)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)], default=5)
    is_featured = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-is_featured', '-created_at']
    
    def __str__(self):
        return f"Testimonial from {self.name}"


class ContactMessage(models.Model):
    """Contact form messages"""
    STATUS_CHOICES = [
        ('new', 'New'),
        ('read', 'Read'),
        ('replied', 'Replied'),
        ('closed', 'Closed'),
    ]
    
    name = models.CharField(max_length=200)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='new')
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    submitted_at = models.DateTimeField(auto_now_add=True)
    replied_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-submitted_at']
    
    def __str__(self):
        return f"{self.name} - {self.subject}"
    
    def mark_as_replied(self):
        self.status = 'replied'
        self.replied_at = timezone.now()
        self.save()


class FAQ(models.Model):
    """Frequently Asked Questions"""
    question = models.CharField(max_length=300)
    answer = models.TextField()
    display_order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['display_order', '-created_at']
        verbose_name = "FAQ"
        verbose_name_plural = "FAQs"
    
    def __str__(self):
        return self.question
    

# ==================== USER ROLE MODEL ====================
class UserRole(models.Model):
    """Role-based access control"""
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('manager', 'Manager'),
        ('storage_staff', 'Storage Staff'),
        ('negotiation_team', 'Negotiation Team'),
        ('client', 'Client'),
        ('vendor', 'Vendor'),  # Adding vendor for consistency
        ('customer', 'Customer'),  # Adding customer for consistency
    ]
    
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='user_role')
    role = models.CharField(max_length=50, choices=ROLE_CHOICES)
    permissions = models.JSONField(default=dict)  # Store granular permissions
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "User Role"
        verbose_name_plural = "User Roles"
        ordering = ['user__username']
    
    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"
    
    def has_permission(self, permission):
        """Check if user has specific permission"""
        return permission in self.permissions.get('allowed', [])


# ==================== INVENTORY MODELS ====================
class InventoryItem(models.Model):
    """Core inventory item model"""
    ITEM_TYPE_CHOICES = [
        ('storage', 'Storage'),
        ('sale', 'For Sale'),
        ('rental', 'Rental'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('reserved', 'Reserved'),
        ('sold', 'Sold'),
        ('expired', 'Expired'),
        ('damaged', 'Damaged'),
        ('returned', 'Returned'),
    ]
    
    # Basic Information
    name = models.CharField(max_length=200)
    sku = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    category = models.ForeignKey('ProductCategory', on_delete=models.CASCADE, related_name='inventory_items')
    product = models.ForeignKey('Product', on_delete=models.SET_NULL, null=True, blank=True, related_name='inventory_items')
    
    # Inventory Details
    quantity = models.IntegerField(default=0)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    total_value = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    low_stock_threshold = models.IntegerField(default=10)
    reorder_point = models.IntegerField(default=20)
    
    # Type & Status
    item_type = models.CharField(max_length=20, choices=ITEM_TYPE_CHOICES, default='storage')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Location & Storage
    location = models.ForeignKey('StorageLocation', on_delete=models.SET_NULL, null=True, blank=True, related_name='inventory_items')
    storage_reference = models.CharField(max_length=100, blank=True)
    batch_number = models.ForeignKey('FruitBatch', on_delete=models.SET_NULL, null=True, blank=True, related_name='inventory_items')

    # Granular Location Details (NEW)
    shelf_number = models.CharField(max_length=50, blank=True, help_text="Specific shelf number within the storage location.")
    bin_number = models.CharField(max_length=50, blank=True, help_text="Specific bin number on the shelf.")
    pallet_id = models.CharField(max_length=100, blank=True, help_text="ID of the pallet the item is on.")
    
    # Time-based
    expiry_date = models.DateField(null=True, blank=True)
    manufactured_date = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Ownership
    client = models.ForeignKey(CustomUser, on_delete=models.CASCADE, 
                             limit_choices_to={'user_type': 'client'},
                             related_name='client_inventory')
    added_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, 
                               related_name='added_inventory_items')
    
    # Quality Information
    quality_rating = models.CharField(max_length=20, choices=[
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('fair', 'Fair'),
        ('poor', 'Poor'),
    ], default='good')
    condition_notes = models.TextField(blank=True)
    
    # Audit fields
    last_checked = models.DateTimeField(null=True, blank=True)
    checked_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, 
                                 related_name='checked_inventory_items')
    next_check_date = models.DateField(null=True, blank=True)
    
    # Dimensions
    weight_kg = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    dimensions = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Inventory Item"
        verbose_name_plural = "Inventory Items"
        indexes = [
            models.Index(fields=['sku', 'status']),
            models.Index(fields=['expiry_date', 'status']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.sku}) - {self.quantity} units"
    
    def save(self, *args, **kwargs):
        # Calculate total value
        self.total_value = self.quantity * self.unit_price
        
        # Generate SKU if not provided
        if not self.sku:
            timestamp = timezone.now().strftime('%Y%m%d%H%M')
            category_prefix = self.category.name[:3].upper() if self.category else 'INV'
            self.sku = f"{category_prefix}-{timestamp}"
        
        # Update last checked if status changed
        if self.pk:
            original = InventoryItem.objects.get(pk=self.pk)
            if original.status != self.status and self.status in ['checked', 'verified']:
                self.last_checked = timezone.now()
        
        super().save(*args, **kwargs)
    
    @property
    def is_low_stock(self):
        """Check if stock is below threshold"""
        return self.quantity <= self.low_stock_threshold
    
    @property
    def needs_reorder(self):
        """Check if needs reorder"""
        return self.quantity <= self.reorder_point
    
    @property
    def is_near_expiry(self):
        """Check if item expires within 30 days"""
        if self.expiry_date:
            days_remaining = (self.expiry_date - timezone.now().date()).days
            return 0 <= days_remaining <= 30
        return False
    
    @property
    def days_until_expiry(self):
        """Calculate days until expiry"""
        if self.expiry_date:
            days = (self.expiry_date - timezone.now().date()).days
            return max(days, 0)
        return None
    
    @property
    def storage_duration(self):
        """Calculate storage duration in days"""
        if self.created_at:
            duration = timezone.now() - self.created_at
            return duration.days
        return 0


class InventoryHistory(models.Model):
    """Track all inventory changes"""
    ACTION_CHOICES = [
        ('create', 'Created'),
        ('update', 'Updated'),
        ('delete', 'Deleted'),
        ('check_in', 'Checked In'),
        ('check_out', 'Checked Out'),
        ('transfer', 'Transferred'),
        ('adjust', 'Adjusted'),
        ('reserve', 'Reserved'),
        ('release', 'Released'),
        ('damage', 'Damaged'),
        ('expire', 'Expired'),
    ]
    
    item = models.ForeignKey(InventoryItem, on_delete=models.CASCADE, related_name='history')
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    user = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, related_name='inventory_actions')
    
    # Changes made
    previous_quantity = models.IntegerField(null=True, blank=True)
    new_quantity = models.IntegerField(null=True, blank=True)
    previous_status = models.CharField(max_length=20, blank=True)
    new_status = models.CharField(max_length=20, blank=True)
    previous_location = models.ForeignKey('StorageLocation', on_delete=models.SET_NULL, null=True, blank=True, related_name='previous_moves')
    new_location = models.ForeignKey('StorageLocation', on_delete=models.SET_NULL, null=True, blank=True, related_name='new_moves')
    
    # Value changes
    previous_unit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    new_unit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    notes = models.TextField(blank=True)
    reference_number = models.CharField(max_length=100, blank=True)  # For linking to orders, deliveries, etc.
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Inventory History"
        verbose_name_plural = "Inventory History"
        indexes = [
            models.Index(fields=['item', 'timestamp']),
            models.Index(fields=['action', 'timestamp']),
        ]
    
    def __str__(self):
        user_name = self.user.username if self.user else 'System'
        return f"{self.item.name} - {self.get_action_display()} by {user_name}"


# ==================== DELIVERY MODELS ====================
class Delivery(models.Model):
    """Delivery tracking system"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('packed', 'Packed'),
        ('in_transit', 'In Transit'),
        ('out_for_delivery', 'Out for Delivery'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
        ('failed', 'Failed'),
        ('returned', 'Returned'),
    ]
    
    PAYMENT_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('partial', 'Partial Payment'),
        ('paid', 'Paid'),
        ('overdue', 'Overdue'),
        ('refunded', 'Refunded'),
    ]
    
    # Delivery Information
    delivery_number = models.CharField(max_length=50, unique=True)
    tracking_number = models.CharField(max_length=100, blank=True, unique=True)
    
    # Related Orders
    order = models.ForeignKey('Order', on_delete=models.SET_NULL, null=True, blank=True, related_name='deliveries')
    
    # Client Information
    client = models.ForeignKey(CustomUser, on_delete=models.CASCADE, 
                             limit_choices_to={'user_type': 'client'},
                             related_name='client_deliveries')
    client_name = models.CharField(max_length=200)
    client_address = models.TextField()
    client_phone = models.CharField(max_length=20)
    client_email = models.EmailField()
    
    # Delivery Details
    delivery_address = models.TextField()
    delivery_city = models.CharField(max_length=100, blank=True)
    delivery_state = models.CharField(max_length=100, blank=True)
    delivery_country = models.CharField(max_length=100, default='Tanzania')
    delivery_postal_code = models.CharField(max_length=20, blank=True)
    
    # Delivery Instructions
    special_instructions = models.TextField(blank=True)
    delivery_type = models.CharField(max_length=20, choices=[
        ('standard', 'Standard'),
        ('express', 'Express'),
        ('sameday', 'Same Day'),
        ('scheduled', 'Scheduled'),
    ], default='standard')
    
    # Time Tracking
    estimated_delivery = models.DateTimeField()
    actual_delivery = models.DateTimeField(null=True, blank=True)
    scheduled_for = models.DateTimeField(null=True, blank=True)
    delivery_window_start = models.TimeField(null=True, blank=True)
    delivery_window_end = models.TimeField(null=True, blank=True)
    
    # Status Tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    status_changed_at = models.DateTimeField(auto_now=True)
    status_changed_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, 
                                        related_name='delivery_status_changes')
    
    # Proof of Delivery
    proof_of_delivery = models.FileField(upload_to='delivery_proofs/%Y/%m/', blank=True, null=True)
    proof_of_delivery_url = models.URLField(blank=True)
    recipient_name = models.CharField(max_length=200, blank=True)
    recipient_phone = models.CharField(max_length=20, blank=True)
    recipient_signature = models.TextField(blank=True)  # Could be base64 encoded signature
    delivery_notes = models.TextField(blank=True)
    delivery_photos = models.JSONField(default=list, blank=True)  # List of photo URLs
    
    # Carrier Integration Details (NEW)
    carrier_name = models.CharField(max_length=100, blank=True, help_text="Name of the shipping carrier (e.g., 'DHL', 'FedEx', 'Local Post').")
    carrier_service_code = models.CharField(max_length=50, blank=True, help_text="Service code or type used by the carrier (e.g., 'EXP', 'STD', 'Priority').")
    external_tracking_url = models.URLField(blank=True, help_text="Direct URL to the carrier's tracking page for this delivery.")

    # Cost & Payment
    delivery_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    delivery_tax = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUS_CHOICES, default='pending')
    payment_method = models.CharField(max_length=50, blank=True)
    
    # Delivery Agent
    assigned_to = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True,
                                  limit_choices_to={'user_type': 'vendor'},
                                  related_name='assigned_deliveries')
    driver_name = models.CharField(max_length=100, blank=True)
    driver_phone = models.CharField(max_length=20, blank=True)
    vehicle_number = models.CharField(max_length=50, blank=True)
    
    # Package Information
    package_count = models.IntegerField(default=0)
    total_weight = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    package_dimensions = models.CharField(max_length=100, blank=True)
    insurance_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    packed_at = models.DateTimeField(null=True, blank=True)
    shipped_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Delivery"
        verbose_name_plural = "Deliveries"
        indexes = [
            models.Index(fields=['delivery_number', 'status']),
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['estimated_delivery', 'status']),
        ]
    
    def __str__(self):
        return f"Delivery #{self.delivery_number} - {self.client_name}"
    
    def save(self, *args, **kwargs):
        if not self.delivery_number:
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            self.delivery_number = f"DEL{timestamp}"
        
        if not self.tracking_number:
            import random
            import string
            random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            self.tracking_number = f"TRK{random_str}"
        
        # Calculate total cost
        self.total_cost = self.delivery_cost + self.delivery_tax
        
        # Update status timestamps
        if self.pk:
            original = Delivery.objects.get(pk=self.pk)
            if original.status != self.status:
                if self.status == 'packed' and not self.packed_at:
                    self.packed_at = timezone.now()
                elif self.status == 'in_transit' and not self.shipped_at:
                    self.shipped_at = timezone.now()
                elif self.status == 'delivered' and not self.actual_delivery:
                    self.actual_delivery = timezone.now()
        
        super().save(*args, **kwargs)
    
    @property
    def is_late(self):
        """Check if delivery is late"""
        if self.status not in ['delivered', 'cancelled', 'failed'] and self.estimated_delivery:
            return timezone.now() > self.estimated_delivery
        return False
    
    @property
    def delivery_duration(self):
        """Calculate delivery duration in hours"""
        if self.actual_delivery and self.created_at:
            duration = self.actual_delivery - self.created_at
            return round(duration.total_seconds() / 3600, 1)
        return None
    
    @property
    def items_total_value(self):
        """Calculate total value of all items in delivery"""
        total = sum(item.total_price for item in self.delivery_items.all())
        return total
    
    def mark_as_delivered(self, recipient_name, signature=None, notes=''):
        """Mark delivery as delivered"""
        self.status = 'delivered'
        self.actual_delivery = timezone.now()
        self.recipient_name = recipient_name
        if signature:
            self.recipient_signature = signature
        self.delivery_notes = notes
        self.save()
    
    def generate_tracking_url(self):
        """Generate tracking URL for the delivery"""
        if self.tracking_number:
            return f"https://bika.com/track/{self.tracking_number}"
        return None


class DeliveryItem(models.Model):
    """Items included in a delivery"""
    delivery = models.ForeignKey(Delivery, on_delete=models.CASCADE, related_name='delivery_items')
    item = models.ForeignKey(InventoryItem, on_delete=models.CASCADE, related_name='delivery_items')
    quantity = models.IntegerField(default=1)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    notes = models.TextField(blank=True)
    
    # Quality at delivery
    delivered_quality = models.CharField(max_length=20, choices=[
        ('excellent', 'Excellent'),
        ('good', 'Good'),
        ('fair', 'Fair'),
        ('poor', 'Poor'),
        ('damaged', 'Damaged'),
    ], blank=True)
    
    class Meta:
        unique_together = ['delivery', 'item']
        verbose_name = "Delivery Item"
        verbose_name_plural = "Delivery Items"
    
    def __str__(self):
        return f"{self.quantity} x {self.item.name} for {self.delivery.delivery_number}"
    
    def save(self, *args, **kwargs):
        # Update inventory quantity when item is added to delivery
        if self.pk:
            original = DeliveryItem.objects.get(pk=self.pk)
            if original.quantity != self.quantity:
                quantity_diff = self.quantity - original.quantity
                self.item.quantity -= quantity_diff
                self.item.save()
        else:
            # New delivery item
            self.item.quantity -= self.quantity
            self.item.save()
        
        super().save(*args, **kwargs)
    
    @property
    def total_price(self):
        return self.quantity * self.unit_price


class DeliveryStatusHistory(models.Model):
    """Track delivery status changes"""
    delivery = models.ForeignKey(Delivery, on_delete=models.CASCADE, related_name='status_history')
    from_status = models.CharField(max_length=20)
    to_status = models.CharField(max_length=20)
    changed_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, related_name='delivery_history_changes')
    notes = models.TextField(blank=True)
    location = models.CharField(max_length=200, blank=True)  # Where status was changed
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Delivery Status History"
        verbose_name_plural = "Delivery Status History"
        indexes = [
            models.Index(fields=['delivery', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.delivery.delivery_number}: {self.from_status} → {self.to_status}"    


class ClientRequest(models.Model):
    """Client requests for services"""
    REQUEST_TYPES = [
        ('storage', 'Storage Request'),
        ('delivery', 'Delivery Request'),
        ('inspection', 'Quality Inspection'),
        ('withdrawal', 'Withdrawal Request'),
        ('other', 'Other Request'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('reviewing', 'Under Review'),
        ('approved', 'Approved'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('rejected', 'Rejected'),
    ]
    
    URGENCY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    # Basic Information
    request_number = models.CharField(max_length=50, unique=True)
    client = models.ForeignKey(CustomUser, on_delete=models.CASCADE, 
                             limit_choices_to={'user_type': 'client'},
                             related_name='client_requests')
    request_type = models.CharField(max_length=20, choices=REQUEST_TYPES)
    
    # Request Details
    title = models.CharField(max_length=200)
    description = models.TextField()
    quantity = models.IntegerField(default=1)
    
    # Related Items
    inventory_items = models.ManyToManyField('InventoryItem', blank=True, related_name='requests')
    
    # Status & Priority
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    urgency = models.CharField(max_length=20, choices=URGENCY_CHOICES, default='medium')
    
    # Dates
    requested_date = models.DateTimeField(auto_now_add=True)
    preferred_delivery_date = models.DateField(null=True, blank=True)
    estimated_completion_date = models.DateField(null=True, blank=True)
    actual_completion_date = models.DateField(null=True, blank=True)
    
    # Assigned Staff
    assigned_to = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True,
                                  related_name='assigned_requests')
    
    # Additional Info
    attachments = models.FileField(upload_to='client_requests/%Y/%m/', blank=True, null=True)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-requested_date']
    
    def __str__(self):
        return f"Request #{self.request_number} - {self.client.username}"
    
    def save(self, *args, **kwargs):
        if not self.request_number:
            import random
            import string
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            random_str = ''.join(random.choices(string.ascii_uppercase, k=3))
            self.request_number = f"REQ-{timestamp}-{random_str}"
        super().save(*args, **kwargs)

# ==================== SHIPPING MODELS ====================
class ShippingMethod(models.Model):
    """Defines available shipping methods and their costs/rules."""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    base_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    # Could add more complex pricing logic later (e.g., per_kg_cost, min_order_value_for_free_shipping)
    is_active = models.BooleanField(default=True)
    estimated_delivery_min_days = models.IntegerField(default=1)
    estimated_delivery_max_days = models.IntegerField(default=7)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['base_cost', 'name']
        verbose_name = "Shipping Method"
        verbose_name_plural = "Shipping Methods"

    def __str__(self):
        return self.name

# ==================== SUPPLIER & PURCHASE ORDER MODELS ====================
class Supplier(models.Model):
    """Represents a product supplier."""
    name = models.CharField(max_length=200, unique=True)
    contact_person = models.CharField(max_length=100, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    city = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, default="Tanzania")
    payment_terms = models.CharField(max_length=100, blank=True, help_text="e.g., Net 30, Due on Receipt")
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name

class PurchaseOrder(models.Model):
    """Represents a purchase order made to a supplier."""
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('pending', 'Pending Approval'),
        ('ordered', 'Ordered'),
        ('partially_received', 'Partially Received'),
        ('received', 'Received'),
        ('cancelled', 'Cancelled'),
        ('returned', 'Returned'),
    ]
    
    order_number = models.CharField(max_length=50, unique=True)
    supplier = models.ForeignKey(Supplier, on_delete=models.CASCADE, related_name='purchase_orders')
    ordered_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_purchase_orders')
    order_date = models.DateTimeField(auto_now_add=True)
    expected_delivery_date = models.DateField(null=True, blank=True)
    actual_delivery_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    total_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-order_date']

    def __str__(self):
        return f"PO-{self.order_number} to {self.supplier.name}"

    def save(self, *args, **kwargs):
        if not self.order_number:
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            self.order_number = f"PO-{timestamp}"
        super().save(*args, **kwargs)

class PurchaseOrderItem(models.Model):
    """Represents a specific item within a purchase order."""
    purchase_order = models.ForeignKey(PurchaseOrder, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='purchase_order_items')
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    # Status for this specific item (e.g., pending, received, backordered)
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('received', 'Received'),
        ('backordered', 'Backordered'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['purchase_order', 'product']
        ordering = ['created_at']

    def __str__(self):
        return f"{self.quantity} x {self.product.name} for PO-{self.purchase_order.order_number}"

    @property
    def total_price(self):
        return self.quantity * self.unit_price

class InventoryTransfer(models.Model):
    """
    Records the movement of inventory items between storage locations.
    Can be initiated manually or automatically based on AI forecasts.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_transit', 'In Transit'),
        ('received', 'Received'),
        ('cancelled', 'Cancelled'),
        ('failed', 'Failed'),
    ]

    transfer_number = models.CharField(max_length=50, unique=True, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='inventory_transfers')
    quantity = models.PositiveIntegerField(help_text="Quantity of product being transferred.")
    
    source_location = models.ForeignKey(StorageLocation, on_delete=models.CASCADE, related_name='outgoing_transfers')
    destination_location = models.ForeignKey(StorageLocation, on_delete=models.CASCADE, related_name='incoming_transfers')
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    requested_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='initiated_transfers')
    approved_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_transfers')
    
    transfer_date = models.DateTimeField(auto_now_add=True)
    shipped_date = models.DateTimeField(null=True, blank=True)
    received_date = models.DateTimeField(null=True, blank=True)
    
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-transfer_date']
        verbose_name = "Inventory Transfer"
        verbose_name_plural = "Inventory Transfers"

    def __str__(self):
        return f"TRF-{self.transfer_number} ({self.product.name}): {self.source_location.name} -> {self.destination_location.name}"

    def save(self, *args, **kwargs):
        if not self.transfer_number:
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            self.transfer_number = f"TRF-{timestamp}"
        super().save(*args, **kwargs)

    # Methods to update inventory during transfer lifecycle
    def mark_as_shipped(self, user=None):
        if self.status == 'pending':
            self.status = 'in_transit'
            self.shipped_date = timezone.now()
            # Reduce stock from source location (if tracking by product/location)
            # This is simplified; real logic would involve specific InventoryItems
            # Update source location occupancy
            if self.source_location.current_occupancy >= self.quantity:
                self.source_location.current_occupancy -= self.quantity
                self.source_location.save()
            else:
                logger.warning(f"Attempted to ship {self.quantity} from {self.source_location.name} but only {self.source_location.current_occupancy} available.")
            
            self.save()
            # Log this in InventoryHistory for relevant InventoryItems if granular
            return True
        return False
    
    def mark_as_received(self, user=None):
        if self.status == 'in_transit':
            self.status = 'received'
            self.received_date = timezone.now()
            # Increase stock at destination location
            self.destination_location.current_occupancy += self.quantity
            self.destination_location.save()
            self.save()
            # Log this in InventoryHistory for relevant InventoryItems
            return True
        return False

class ProductPriceHistory(models.Model):
    """Tracks historical price changes for a product."""
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='price_history')
    old_price = models.DecimalField(max_digits=10, decimal_places=2)
    new_price = models.DecimalField(max_digits=10, decimal_places=2)
    change_date = models.DateTimeField(auto_now_add=True)
    changed_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True)
    reason = models.TextField(blank=True, help_text="Reason for the price change (e.g., 'Automated AI adjustment', 'Manual adjustment', 'Promotion')")

    class Meta:
        ordering = ['-change_date']
        verbose_name = "Product Price History"
        verbose_name_plural = "Product Price History"

    def __str__(self):
        return f"Price change for {self.product.name}: {self.old_price} -> {self.new_price} on {self.change_date.strftime('%Y-%m-%d')}"

class AIPredictionAccuracy(models.Model):
    """Tracks the accuracy of AI predictions over time."""
    PREDICTION_TYPE_CHOICES = [
        ('stock_level', 'Stock Level Prediction'),
        ('demand_forecast', 'Demand Forecast'),
        ('quality_class', 'Quality Classification'),
        ('expiry_date', 'Expiry Date Prediction'),
        # Add other prediction types as needed
    ]

    model = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='accuracy_records')
    prediction_type = models.CharField(max_length=50, choices=PREDICTION_TYPE_CHOICES)
    evaluation_date = models.DateTimeField(auto_now_add=True)
    
    metric_name = models.CharField(max_length=50, help_text="e.g., MAE, RMSE, Accuracy, F1-Score")
    metric_value = models.FloatField()
    
    period_start = models.DateField(help_text="Start date of the prediction period.")
    period_end = models.DateField(help_text="End date of the prediction period.")
    
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True, help_text="Specific product if evaluation is product-specific.")
    # Could also add location if predictions are location-specific

    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-evaluation_date']
        verbose_name = "AI Prediction Accuracy"
        verbose_name_plural = "AI Prediction Accuracy"

    def __str__(self):
        return f"{self.prediction_type} ({self.metric_name}: {self.metric_value:.4f}) for {self.model.name if self.model else 'N/A'}"

class InventoryMovement(models.Model):
    """Logs detailed movements of individual inventory items."""
    MOVEMENT_TYPE_CHOICES = [
        ('receive', 'Receive into Location'),
        ('put_away', 'Put Away (to shelf/bin)'),
        ('pick', 'Pick from Shelf/Bin'),
        ('pack', 'Pack for Shipment'),
        ('transfer_in', 'Transfer In (from another location)'),
        ('transfer_out', 'Transfer Out (to another location)'),
        ('adjust_in', 'Adjustment In'),
        ('adjust_out', 'Adjustment Out'),
        ('return', 'Customer Return'),
        ('damage', 'Damage/Spoilage'),
    ]

    item = models.ForeignKey(InventoryItem, on_delete=models.CASCADE, related_name='movements')
    movement_type = models.CharField(max_length=50, choices=MOVEMENT_TYPE_CHOICES)
    
    # Locations
    from_location = models.ForeignKey(StorageLocation, on_delete=models.SET_NULL, null=True, blank=True, related_name='outgoing_movements')
    to_location = models.ForeignKey(StorageLocation, on_delete=models.SET_NULL, null=True, blank=True, related_name='incoming_movements')
    
    # Granular locations (within the StorageLocation)
    from_shelf = models.CharField(max_length=50, blank=True)
    from_bin = models.CharField(max_length=50, blank=True)
    from_pallet = models.CharField(max_length=100, blank=True)

    to_shelf = models.CharField(max_length=50, blank=True)
    to_bin = models.CharField(max_length=50, blank=True)
    to_pallet = models.CharField(max_length=100, blank=True)

    quantity_moved = models.PositiveIntegerField(default=1) # How many units of this item were moved
    
    moved_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='inventory_movements')
    movement_date = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-movement_date']
        verbose_name = "Inventory Movement Log"
        verbose_name_plural = "Inventory Movement Logs"

    def __str__(self):
        return f"{self.item.name} ({self.quantity_moved}x) moved {self.movement_type} by {self.moved_by.username if self.moved_by else 'System'}"

class Coupon(models.Model):
    """Represents a discount coupon."""
    DISCOUNT_TYPE_CHOICES = [
        ('percentage', 'Percentage Discount'),
        ('fixed_amount', 'Fixed Amount Discount'),
        ('free_shipping', 'Free Shipping'),
    ]

    code = models.CharField(max_length=50, unique=True, help_text="Unique code for the coupon.")
    description = models.TextField(blank=True, help_text="Description of the coupon (e.g., '10% off all fruits').")
    
    discount_type = models.CharField(max_length=20, choices=DISCOUNT_TYPE_CHOICES)
    value = models.DecimalField(max_digits=10, decimal_places=2, help_text="Value of the discount (e.g., 10 for 10%, 5.00 for $5 off).")
    
    is_active = models.BooleanField(default=True)
    valid_from = models.DateTimeField(default=timezone.now)
    valid_to = models.DateTimeField(null=True, blank=True)
    
    usage_limit = models.PositiveIntegerField(null=True, blank=True, help_text="Maximum number of times this coupon can be used overall.")
    used_count = models.PositiveIntegerField(default=0)
    
    # Optional rules
    min_purchase_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Minimum purchase amount required to use this coupon.")
    
    # Apply to specific products or categories (optional)
    applicable_products = models.ManyToManyField(Product, blank=True, related_name='coupons')
    applicable_categories = models.ManyToManyField(ProductCategory, blank=True, related_name='coupons')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Coupon"
        verbose_name_plural = "Coupons"

    def __str__(self):
        return self.code

    def is_valid(self, cart_total=None, user_usage_count=0):
        """Checks if the coupon is currently valid."""
        now = timezone.now()
        if not self.is_active or (self.valid_to and now > self.valid_to) or now < self.valid_from:
            return False, "Coupon is not active or expired."
        
        if self.usage_limit and self.used_count >= self.usage_limit:
            return False, "Coupon has reached its maximum usage limit."
        
        # User-specific usage limit could be tracked in a separate model if needed
        # if self.per_user_usage_limit and user_usage_count >= self.per_user_usage_limit:
        #     return False, "You have already used this coupon the maximum number of times."

        if self.min_purchase_amount and cart_total is not None and cart_total < self.min_purchase_amount:
            return False, f"Minimum purchase amount of {self.min_purchase_amount} is required."
            
        return True, "Coupon is valid."

# ==================== PRODUCT AI INSIGHTS MODEL ====================
class ProductAIInsights(models.Model):
    """Stores AI-derived predictions and insights for a product."""

    QUALITY_CLASSES = [
        ('Excellent', 'Excellent'),
        ('Good', 'Good'),
        ('Fair', 'Fair'),
        ('Poor', 'Poor'),
        ('Critical', 'Critical'), # General term for Rotten/Unsellable
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='ai_insights_for_product', null=True, blank=True, help_text="The general product definition (optional if inventory_item is set).")
    inventory_item = models.OneToOneField('InventoryItem', on_delete=models.CASCADE, related_name='ai_insights', null=True, blank=True, help_text="The specific inventory item instance (if insight is item-specific).")
    storage_location = models.ForeignKey('StorageLocation', on_delete=models.SET_NULL, related_name='ai_insights_in_location', null=True, blank=True, help_text="The storage location related to this insight.")
    
    # Stock Predictions
    predicted_stock_level = models.IntegerField(default=0, help_text="AI's predicted stock level in the near future.")
    predicted_out_of_stock_date = models.DateField(null=True, blank=True, help_text="AI's predicted date when product might run out.")
    
    # Expiry/Quality Predictions (general for any perishable, not just fruit)
    predicted_expiry_date = models.DateField(null=True, blank=True, help_text="AI's predicted expiry date for the product.")
    predicted_quality_class = models.CharField(max_length=20, choices=QUALITY_CLASSES, blank=True, null=True, help_text="AI's predicted quality class.")
    
    # Demand Forecasting
    demand_forecast_next_7_days = models.IntegerField(default=0, help_text="AI's forecasted demand for the next 7 days.")
    demand_forecast_next_30_days = models.IntegerField(default=0, help_text="AI's forecasted demand for the next 30 days.")

    # General AI Analysis Metrics
    prediction_confidence = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, help_text="Confidence score of the latest prediction.")
    
    # Audit Fields
    last_analyzed = models.DateTimeField(auto_now=True, help_text="Timestamp of the last AI analysis.")
    analysis_model_used = models.ForeignKey('TrainedModel', on_delete=models.SET_NULL, null=True, blank=True, help_text="The AI model used for the last analysis.")

    class Meta:
        verbose_name = "Product AI Insight"
        verbose_name_plural = "Product AI Insights"
        indexes = [
            models.Index(fields=['product']),
            models.Index(fields=['inventory_item']),
            models.Index(fields=['storage_location']),
            models.Index(fields=['predicted_out_of_stock_date']),
            models.Index(fields=['predicted_expiry_date']),
            models.Index(fields=['predicted_quality_class']),
        ]
        # An inventory item can only have one insight.
        # A product at a specific storage location can also have a unique insight.
        unique_together = (('inventory_item',), ('product', 'storage_location')) 
        
    def __str__(self):
        if self.inventory_item:
            return f"AI Insights for Inventory Item: {self.inventory_item.name} at {self.inventory_item.location.name}"
        elif self.product and self.storage_location:
            return f"AI Insights for Product: {self.product.name} at {self.storage_location.name}"
        elif self.product:
            return f"AI Insights for Product: {self.product.name}"
        return f"AI Insights ({self.pk})"