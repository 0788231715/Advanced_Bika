from django.db import models
from bika.models import CustomUser

class ProductCategory(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Product Categories"

    def __str__(self):
        return self.name

class Product(models.Model):
    owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='products')
    category = models.ForeignKey(ProductCategory, on_delete=models.SET_NULL, null=True, blank=True, related_name='products')
    name = models.CharField(max_length=200)
    description = models.TextField()
    image = models.ImageField(upload_to='products/', blank=True, null=True)
    
    # Price for the end customer
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Charges for the client/owner
    storage_charges = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    client_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, 
                                     help_text="Price charged to the client for this product, if applicable.")
    
    # Status fields
    is_approved = models.BooleanField(default=False, 
                                      help_text="Whether this product has been approved by an admin.")
    is_available = models.BooleanField(default=True, 
                                       help_text="Whether this product is currently available for sale.")
    views_count = models.PositiveIntegerField(default=0) # Assuming this was intended from the folder structure `0004_product_views_count.py`
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name