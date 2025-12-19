# bika/signals.py
from django.db.models.signals import post_save, pre_save, post_delete, m2m_changed
from django.dispatch import receiver
from django.utils import timezone
from django.contrib.auth.models import Group
from django.db import transaction

from .models import (
    CustomUser, Product, ProductAlert, FruitBatch, 
    FruitQualityReading, Order, OrderItem, Cart,
    ProductReview, Wishlist, RealTimeSensorData
)
from .services.ai_service import enhanced_ai_service
from bika import models

# ==================== USER SIGNALS ====================

@receiver(post_save, sender=CustomUser)
def handle_user_creation(sender, instance, created, **kwargs):
    """Handle user creation and assign permissions"""
    if created:
        try:
            # Assign to appropriate group based on user type
            if instance.user_type == 'vendor':
                vendor_group, _ = Group.objects.get_or_create(name='Vendors')
                instance.groups.add(vendor_group)
                
                # Create welcome notification
                from .models import Notification
                Notification.objects.create(
                    user=instance,
                    title="Welcome to Bika Vendor Portal!",
                    message=f"Welcome {instance.business_name or instance.username}! Get started by adding your first product.",
                    notification_type='system_alert'
                )
                
            elif instance.user_type == 'customer':
                customer_group, _ = Group.objects.get_or_create(name='Customers')
                instance.groups.add(customer_group)
                
        except Exception as e:
            print(f"Error in user creation signal: {e}")

# ==================== PRODUCT SIGNALS ====================

@receiver(pre_save, sender=Product)
def handle_product_save(sender, instance, **kwargs):
    """Handle product pre-save operations"""
    if not instance.slug:
        from django.utils.text import slugify
        instance.slug = slugify(instance.name)
    
    if not instance.sku:
        import random
        instance.sku = f"PROD{timezone.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
    
    # Auto-publish if status changes to active and not published
    if instance.status == 'active' and not instance.published_at:
        instance.published_at = timezone.now()
    
    # Generate barcode if not exists
    if not instance.barcode and instance.track_inventory:
        import random
        instance.barcode = f"8{random.randint(100000000000, 999999999999)}"

@receiver(post_save, sender=Product)
def handle_product_post_save(sender, instance, created, **kwargs):
    """Handle product post-save operations"""
    if created:
        # Create welcome alert for vendor
        ProductAlert.objects.create(
            product=instance,
            alert_type='stock_low',
            severity='low',
            message=f"New product '{instance.name}' created. Add inventory to start selling.",
            detected_by='system'
        )
    
    # Check for low stock alerts
    if instance.track_inventory:
        if instance.stock_quantity <= 0:
            ProductAlert.objects.get_or_create(
                product=instance,
                alert_type='stock_low',
                severity='critical',
                message=f"Product '{instance.name}' is out of stock!",
                detected_by='system',
                is_resolved=False
            )
        elif instance.stock_quantity <= instance.low_stock_threshold:
            ProductAlert.objects.get_or_create(
                product=instance,
                alert_type='stock_low',
                severity='medium',
                message=f"Product '{instance.name}' is low on stock ({instance.stock_quantity} left).",
                detected_by='system',
                is_resolved=False
            )

# ==================== FRUIT BATCH SIGNALS ====================

@receiver(post_save, sender=FruitBatch)
def handle_fruit_batch_creation(sender, instance, created, **kwargs):
    """Handle fruit batch creation"""
    if created:
        # Create initial quality reading
        FruitQualityReading.objects.create(
            fruit_batch=instance,
            temperature=5.0,
            humidity=90.0,
            light_intensity=50.0,
            co2_level=400,
            predicted_class='Good',
            confidence_score=0.85,
            notes="Initial reading after batch creation"
        )
        
        # Create monitoring alert
        ProductAlert.objects.create(
            product=instance.product if instance.product else None,
            alert_type='quality_issue',
            severity='low',
            message=f"New fruit batch {instance.batch_number} created. Monitoring started.",
            detected_by='system'
        )

@receiver(pre_save, sender=FruitBatch)
def handle_fruit_batch_expiry(sender, instance, **kwargs):
    """Check fruit batch expiry"""
    if instance.expected_expiry:
        days_remaining = (instance.expected_expiry - timezone.now()).days
        
        if days_remaining <= 2 and instance.status == 'active':
            # Update status to critical
            instance.status = 'active'
            
            # Create expiry alert
            ProductAlert.objects.create(
                product=instance.product if instance.product else None,
                alert_type='expiry_near',
                severity='critical',
                message=f"Fruit batch {instance.batch_number} expires in {days_remaining} days!",
                detected_by='system'
            )

# ==================== QUALITY READING SIGNALS ====================

@receiver(post_save, sender=FruitQualityReading)
def handle_quality_reading(sender, instance, created, **kwargs):
    """Handle quality reading creation"""
    if created:
        # Check for quality issues
        if instance.predicted_class in ['Poor', 'Rotten']:
            ProductAlert.objects.create(
                product=instance.fruit_batch.product if instance.fruit_batch.product else None,
                alert_type='quality_issue',
                severity='high',
                message=f"Poor quality detected in batch {instance.fruit_batch.batch_number}: {instance.predicted_class}",
                detected_by='ai_system'
            )
        
        # Check for temperature anomalies
        if instance.temperature < 0 or instance.temperature > 15:
            ProductAlert.objects.create(
                product=instance.fruit_batch.product if instance.fruit_batch.product else None,
                alert_type='temperature_anomaly',
                severity='medium',
                message=f"Temperature anomaly detected in batch {instance.fruit_batch.batch_number}: {instance.temperature}°C",
                detected_by='sensor_system'
            )
        
        # Update batch status based on quality
        try:
            if instance.predicted_class == 'Rotten' and instance.fruit_batch.status == 'active':
                instance.fruit_batch.status = 'discarded'
                instance.fruit_batch.save()
        except:
            pass

# ==================== ORDER SIGNALS ====================

@receiver(post_save, sender=Order)
def handle_order_status_change(sender, instance, **kwargs):
    """Handle order status changes"""
    # Create notification for user
    if not kwargs.get('created', False):  # Only on updates
        from .models import Notification
        
        status_messages = {
            'confirmed': f"Your order #{instance.order_number} has been confirmed.",
            'shipped': f"Your order #{instance.order_number} has been shipped.",
            'delivered': f"Your order #{instance.order_number} has been delivered.",
            'cancelled': f"Your order #{instance.order_number} has been cancelled.",
        }
        
        if instance.status in status_messages:
            Notification.objects.create(
                user=instance.user,
                title=f"Order #{instance.order_number} Update",
                message=status_messages[instance.status],
                notification_type='order_update',
                related_object_type='order',
                related_object_id=instance.id
            )

@receiver(post_save, sender=OrderItem)
def handle_order_item_save(sender, instance, created, **kwargs):
    """Handle order item creation"""
    if created and instance.order.status == 'delivered':
        # Update product stock
        if instance.product.track_inventory:
            instance.product.stock_quantity -= instance.quantity
            instance.product.save()

# ==================== REVIEW SIGNALS ====================

@receiver(post_save, sender=ProductReview)
def handle_review_creation(sender, instance, created, **kwargs):
    """Handle product review creation"""
    if created:
        # Update product average rating
        reviews = ProductReview.objects.filter(
            product=instance.product,
            is_approved=True
        )
        avg_rating = reviews.aggregate(models.Avg('rating'))['rating__avg']
        
        if avg_rating:
            # You could store this in a cache or denormalized field
            pass
        
        # Create notification for vendor
        if instance.product.vendor != instance.user:
            from .models import Notification
            Notification.objects.create(
                user=instance.product.vendor,
                title="New Product Review",
                message=f"{instance.user.username} reviewed your product '{instance.product.name}'",
                notification_type='product_alert',
                related_object_type='product_review',
                related_object_id=instance.id
            )

# ==================== CART SIGNALS ====================

@receiver(post_delete, sender=Cart)
def handle_cart_removal(sender, instance, **kwargs):
    """Handle cart item removal"""
    # Check if product is back in stock
    if instance.product.track_inventory and instance.product.stock_quantity == 0:
        # Check if there's an out-of-stock alert
        alert = ProductAlert.objects.filter(
            product=instance.product,
            alert_type='stock_low',
            severity='critical',
            is_resolved=False
        ).first()
        
        if alert:
            alert.is_resolved = True
            alert.resolved_at = timezone.now()
            alert.save()

# ==================== SENSOR DATA SIGNALS ====================

@receiver(post_save, sender=RealTimeSensorData)
def handle_sensor_data(sender, instance, created, **kwargs):
    """Handle real-time sensor data"""
    if created:
        # Check for anomalies based on sensor type
        if instance.sensor_type == 'temperature':
            if instance.value < 0 or instance.value > 15:
                ProductAlert.objects.create(
                    product=instance.product,
                    alert_type='temperature_anomaly',
                    severity='medium',
                    message=f"Temperature anomaly detected: {instance.value}{instance.unit}",
                    detected_by='sensor_system'
                )
        
        elif instance.sensor_type == 'humidity':
            if instance.value < 80 or instance.value > 100:
                ProductAlert.objects.create(
                    product=instance.product,
                    alert_type='humidity_issue',
                    severity='low',
                    message=f"Humidity anomaly detected: {instance.value}{instance.unit}",
                    detected_by='sensor_system'
                )
        
        elif instance.sensor_type == 'co2':
            if instance.value > 1000:
                ProductAlert.objects.create(
                    product=instance.product,
                    alert_type='ai_anomaly',
                    severity='medium',
                    message=f"High CO2 level detected: {instance.value}{instance.unit}. Improve ventilation.",
                    detected_by='sensor_system'
                )

# ==================== ALERT RESOLUTION SIGNALS ====================

@receiver(pre_save, sender=ProductAlert)
def handle_alert_resolution(sender, instance, **kwargs):
    """Handle alert resolution"""
    # If alert is being resolved and wasn't resolved before
    if instance.is_resolved and not instance.pk:
        old_instance = ProductAlert.objects.get(pk=instance.pk) if instance.pk else None
        if old_instance and not old_instance.is_resolved:
            instance.resolved_at = timezone.now()

# ==================== WISHLIST SIGNALS ====================

@receiver(post_save, sender=Wishlist)
def handle_wishlist_addition(sender, instance, created, **kwargs):
    """Handle wishlist item addition"""
    if created:
        # Notify vendor if customer added their product to wishlist
        if instance.product.vendor != instance.user:
            from .models import Notification
            Notification.objects.create(
                user=instance.product.vendor,
                title="Product Added to Wishlist",
                message=f"{instance.user.username} added your product '{instance.product.name}' to their wishlist",
                notification_type='product_alert',
                related_object_type='wishlist',
                related_object_id=instance.id
            )

# ==================== CONNECT SIGNALS ====================

def connect_signals():
    """Connect all signals"""
    # User signals
    post_save.connect(handle_user_creation, sender=CustomUser)
    
    # Product signals
    pre_save.connect(handle_product_save, sender=Product)
    post_save.connect(handle_product_post_save, sender=Product)
    
    # Fruit monitoring signals
    pre_save.connect(handle_fruit_batch_expiry, sender=FruitBatch)
    post_save.connect(handle_fruit_batch_creation, sender=FruitBatch)
    post_save.connect(handle_quality_reading, sender=FruitQualityReading)
    
    # Order signals
    post_save.connect(handle_order_status_change, sender=Order)
    post_save.connect(handle_order_item_save, sender=OrderItem)
    
    # Review signals
    post_save.connect(handle_review_creation, sender=ProductReview)
    
    # Cart signals
    post_delete.connect(handle_cart_removal, sender=Cart)
    
    # Sensor signals
    post_save.connect(handle_sensor_data, sender=RealTimeSensorData)
    
    # Alert signals
    pre_save.connect(handle_alert_resolution, sender=ProductAlert)
    
    # Wishlist signals
    post_save.connect(handle_wishlist_addition, sender=Wishlist)

# Connect signals when Django starts
connect_signals()
print("Bika: Signals connected successfully")