from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    """Custom user model with different user types"""
    USER_TYPE_CHOICES = [
        ('customer', 'Customer'),
        ('vendor', 'Vendor'),
        ('admin', 'Administrator'),
    ]
    
    user_type = models.CharField(max_length=20, choices=USER_TYPE_CHOICES, default='customer')
    phone = models.CharField(max_length=20, blank=True)
    company = models.CharField(max_length=100, blank=True)
    address = models.TextField(blank=True)
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    email_verified = models.BooleanField(default=False)
    phone_verified = models.BooleanField(default=False)
    
    # Additional fields for vendors
    business_name = models.CharField(max_length=200, blank=True)
    business_description = models.TextField(blank=True)
    business_logo = models.ImageField(upload_to='business_logos/', blank=True, null=True)
    business_verified = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.username} ({self.get_user_type_display()})"
    
    def is_vendor(self):
        return self.user_type == 'vendor'
    
    def is_customer(self):
        return self.user_type == 'customer'