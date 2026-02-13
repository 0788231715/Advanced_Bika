# bika/forms.py - COMPLETE AND COMBINED VERSION
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone
import json

# Import ALL models from your application
from .models import (
    CustomUser, SiteInfo, Service, ProductCategory, Product, 
    ProductImage, ProductReview, Wishlist, Cart, Order, OrderItem, 
    Payment, ContactMessage, FAQ, StorageLocation, FruitType, 
    FruitBatch, FruitQualityReading, RealTimeSensorData, 
    ProductAlert, Notification, ProductDataset, TrainedModel,
    PaymentGatewaySettings, CurrencyExchangeRate, Testimonial,
    InventoryItem, Delivery, DeliveryItem, DeliveryStatusHistory,
    InventoryHistory, ClientRequest, UserRole, Address,
    NotificationSettings, TwoFactorSettings
)

User = get_user_model()

# ==================== AUTHENTICATION FORMS ====================

class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Username or Email'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password'
        })
    )
    remember_me = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your email address'
        })
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'First Name'
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Last Name'
        })
    )
    phone = forms.CharField(
        max_length=20,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Phone Number'
        })
    )
    user_type = forms.ChoiceField(
        choices=CustomUser.USER_TYPE_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'phone', 'user_type', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Style the existing fields
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Choose a username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm password'
        })
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if email and User.objects.filter(email=email).exists():
            raise ValidationError("A user with this email already exists.")
        return email
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.phone = self.cleaned_data['phone']
        user.user_type = self.cleaned_data['user_type']
        if commit:
            user.save()
        return user

class VendorRegistrationForm(CustomUserCreationForm):
    business_name = forms.CharField(
        max_length=200,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Your Business Name'
        })
    )
    business_description = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Describe your business',
            'rows': 3
        }),
        required=False
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'phone', 'business_name', 'business_description', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set user_type to vendor and hide it
        self.fields['user_type'].initial = 'vendor'
        self.fields['user_type'].widget = forms.HiddenInput()
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.user_type = 'vendor'
        user.business_name = self.cleaned_data['business_name']
        user.business_description = self.cleaned_data['business_description']
        if commit:
            user.save()
        return user

class CustomerRegistrationForm(CustomUserCreationForm):
    agree_terms = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'phone', 'password1', 'password2', 'agree_terms')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set user_type to customer and hide it
        self.fields['user_type'].initial = 'customer'
        self.fields['user_type'].widget = forms.HiddenInput()
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.user_type = 'customer'
        if commit:
            user.save()
        return user

# ==================== USER PROFILE FORMS ====================

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'phone', 'company', 'address', 'profile_picture')
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'company': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'}),
        }

class VendorProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'phone', 'business_name', 'business_description', 'business_logo', 'address')
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'business_name': forms.TextInput(attrs={'class': 'form-control'}),
            'business_description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'business_logo': forms.FileInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

# ==================== ADDRESS FORM ====================

class AddressForm(forms.ModelForm):
    """Form for creating/editing addresses"""
    class Meta:
        model = Address
        fields = [
            'title', 'full_name', 'phone_number', 'street_address',
            'city', 'state', 'postal_code', 'country',
            'is_default_shipping', 'is_default_billing'
        ]
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Home, Work'}),
            'full_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name'}),
            'phone_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Phone Number'}),
            'street_address': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Street Address'}),
            'city': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'City'}),
            'state': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'State/Province'}),
            'postal_code': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Postal Code'}),
            'country': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Country'}),
            'is_default_shipping': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_default_billing': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        if self.user:
            instance.user = self.user
        if commit:
            instance.save()
        return instance

# ==================== PRODUCT FORMS ====================

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = [
            'name', 'slug', 'sku', 'category', 'description', 'short_description', 'tags',
            'price', 'compare_price', 'cost_price', 'tax_rate',
            'stock_quantity', 'low_stock_threshold', 'track_inventory', 'allow_backorders',
            'brand', 'model', 'weight', 'dimensions', 'color', 'size', 'material',
            'status', 'condition', 'is_featured', 'is_digital',
            'meta_title', 'meta_description'
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Product Name'}),
            'slug': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'product-slug'}),
            'sku': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'SKU001'}),
            'category': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Detailed product description'}),
            'short_description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2, 'placeholder': 'Brief product description'}),
            'tags': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'tag1, tag2, tag3'}),
            'price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': '0.00'}),
            'compare_price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': '0.00'}),
            'cost_price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': '0.00'}),
            'tax_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': '0.00'}),
            'stock_quantity': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0'}),
            'low_stock_threshold': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '5'}),
            'track_inventory': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'allow_backorders': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'brand': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Brand Name'}),
            'model': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Model Number'}),
            'weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': 'Weight in kg'}),
            'dimensions': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '10x5x2'}),
            'color': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Red, Blue, etc.'}),
            'size': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'S, M, L, XL'}),
            'material': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Cotton, Plastic, etc.'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
            'condition': forms.Select(attrs={'class': 'form-control'}),
            'is_featured': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_digital': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'meta_title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'SEO Meta Title'}),
            'meta_description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2, 'placeholder': 'SEO Meta Description'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial values for new products
        if not self.instance.pk:
            self.fields['status'].initial = 'draft'
            self.fields['condition'].initial = 'new'
            self.fields['track_inventory'].initial = True
            self.fields['tax_rate'].initial = 0.0
    
    def clean_sku(self):
        sku = self.cleaned_data.get('sku')
        if sku and Product.objects.filter(sku=sku).exclude(pk=self.instance.pk).exists():
            raise ValidationError("A product with this SKU already exists.")
        return sku
    
    def clean_price(self):
        price = self.cleaned_data.get('price')
        if price and price < 0:
            raise ValidationError("Price cannot be negative.")
        return price
    
    def clean_compare_price(self):
        compare_price = self.cleaned_data.get('compare_price')
        price = self.cleaned_data.get('price')
        
        if compare_price and price and compare_price <= price:
            raise ValidationError("Compare price must be greater than the current price.")
        return compare_price
    
    def clean_stock_quantity(self):
        stock_quantity = self.cleaned_data.get('stock_quantity')
        if stock_quantity and stock_quantity < 0:
            raise ValidationError("Stock quantity cannot be negative.")
        return stock_quantity

class ClientProductCreationForm(forms.ModelForm):
    owner = forms.ModelChoiceField(
        queryset=CustomUser.objects.filter(user_type='customer'),
        label="Client/Customer",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = Product
        fields = [
            'owner', 'name', 'category', 'description', 'image',
            'price', 'storage_charges', 'client_price',
            'stock_quantity', 'is_approved', 'is_available'
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Product Name'}),
            'category': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Detailed product description'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': 'Selling Price'}),
            'storage_charges': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': 'Storage Charges per Unit'}),
            'client_price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': 'Price Charged to Client'}),
            'stock_quantity': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0'}),
            'is_approved': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial values for new products
        if not self.instance.pk:
            self.fields['is_approved'].initial = True
            self.fields['is_available'].initial = True
            self.fields['storage_charges'].initial = 0.00
            self.fields['client_price'].initial = 0.00
        
        # Populate category choices
        self.fields['category'].queryset = ProductCategory.objects.filter(is_active=True)

    def clean_price(self):
        price = self.cleaned_data.get('price')
        if price is not None and price < 0:
            raise ValidationError("Selling price cannot be negative.")
        return price

    def clean_storage_charges(self):
        storage_charges = self.cleaned_data.get('storage_charges')
        if storage_charges is not None and storage_charges < 0:
            raise ValidationError("Storage charges cannot be negative.")
        return storage_charges

    def clean_client_price(self):
        client_price = self.cleaned_data.get('client_price')
        if client_price is not None and client_price < 0:
            raise ValidationError("Client price cannot be negative.")
        return client_price

    def clean_stock_quantity(self):
        stock_quantity = self.cleaned_data.get('stock_quantity')
        if stock_quantity is not None and stock_quantity < 0:
            raise ValidationError("Stock quantity cannot be negative.")
        return stock_quantity

class ProductImageForm(forms.ModelForm):
    class Meta:
        model = ProductImage
        fields = ['image', 'alt_text', 'display_order', 'is_primary']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'alt_text': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Description of the image'}),
            'display_order': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_primary': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class ProductImageInlineForm(forms.ModelForm):
    class Meta:
        model = ProductImage
        fields = ['image', 'alt_text', 'display_order', 'is_primary']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'alt_text': forms.TextInput(attrs={'class': 'form-control'}),
            'display_order': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_primary': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class ProductReviewForm(forms.ModelForm):
    class Meta:
        model = ProductReview
        fields = ['rating', 'title', 'comment']
        widgets = {
            'rating': forms.Select(attrs={'class': 'form-control'}),
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Review title'}),
            'comment': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Share your experience with this product'}),
        }
    
    def clean_rating(self):
        rating = self.cleaned_data.get('rating')
        if rating not in [1, 2, 3, 4, 5]:
            raise ValidationError("Please select a valid rating.")
        return rating

# ==================== CATEGORY FORMS ====================

class ProductCategoryForm(forms.ModelForm):
    class Meta:
        model = ProductCategory
        fields = ['name', 'slug', 'description', 'image', 'display_order', 'is_active', 'parent']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'slug': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'display_order': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'parent': forms.Select(attrs={'class': 'form-control'}),
        }

# ==================== SEARCH & FILTER FORMS ====================

class ProductSearchForm(forms.Form):
    query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search products...',
            'name': 'q'
        })
    )
    category = forms.ModelChoiceField(
        queryset=ProductCategory.objects.filter(is_active=True),
        required=False,
        empty_label="All Categories",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    min_price = forms.DecimalField(
        required=False,
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Min Price',
            'step': '0.01'
        })
    )
    max_price = forms.DecimalField(
        required=False,
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Max Price',
            'step': '0.01'
        })
    )
    condition = forms.ChoiceField(
        choices=[('', 'Any Condition')] + Product.CONDITION_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class ProductFilterForm(forms.Form):
    SORT_CHOICES = [
        ('newest', 'Newest First'),
        ('price_low', 'Price: Low to High'),
        ('price_high', 'Price: High to Low'),
        ('name_asc', 'Name: A to Z'),
        ('name_desc', 'Name: Z to A'),
        ('rating', 'Highest Rated'),
    ]
    
    sort_by = forms.ChoiceField(
        choices=SORT_CHOICES,
        required=False,
        initial='newest',
        widget=forms.Select(attrs={'class': 'form-control', 'onchange': 'this.form.submit()'})
    )
    in_stock = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'onchange': 'this.form.submit()'})
    )
    featured = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'onchange': 'this.form.submit()'})
    )

# ==================== CART & ORDER FORMS ====================

class CartItemForm(forms.ModelForm):
    quantity = forms.IntegerField(
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'style': 'width: 80px;'
        })
    )
    
    class Meta:
        model = Cart
        fields = ['quantity']

class CheckoutForm(forms.Form):
    shipping_address = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your shipping address',
            'rows': 3
        })
    )
    billing_address = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your billing address (leave blank if same as shipping)',
            'rows': 3
        })
    )
    payment_method = forms.ChoiceField(
        choices=Payment.PAYMENT_METHODS,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    phone_number = forms.CharField(
        required=False,
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Phone number for mobile money payments'
        })
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Any special instructions?',
            'rows': 2
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        payment_method = cleaned_data.get('payment_method')
        phone_number = cleaned_data.get('phone_number')
        
        # If mobile money payment, phone number is required
        if payment_method in ['mpesa', 'tigo_tz', 'airtel_tz', 'mtn_rw', 'airtel_rw', 'mtn_ug', 'airtel_ug', 'mpesa_ke']:
            if not phone_number:
                raise ValidationError("Phone number is required for mobile money payments.")
        
        return cleaned_data

# ==================== FRUIT QUALITY MONITORING FORMS ====================

class FruitTypeForm(forms.ModelForm):
    class Meta:
        model = FruitType
        fields = ['name', 'scientific_name', 'image', 'description',
                 'optimal_temp_min', 'optimal_temp_max',
                 'optimal_humidity_min', 'optimal_humidity_max',
                 'optimal_light_max', 'optimal_co2_max',
                 'shelf_life_days', 'ethylene_sensitive', 'chilling_sensitive']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'scientific_name': forms.TextInput(attrs={'class': 'form-control'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'optimal_temp_min': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'optimal_temp_max': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'optimal_humidity_min': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'optimal_humidity_max': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'optimal_light_max': forms.NumberInput(attrs={'class': 'form-control'}),
            'optimal_co2_max': forms.NumberInput(attrs={'class': 'form-control'}),
            'shelf_life_days': forms.NumberInput(attrs={'class': 'form-control'}),
            'ethylene_sensitive': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'chilling_sensitive': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class FruitBatchForm(forms.ModelForm):
    expected_expiry = forms.DateTimeField(
        widget=forms.DateTimeInput(
            attrs={'class': 'form-control', 'type': 'datetime-local'},
            format='%Y-%m-%dT%H:%M'
        ),
        input_formats=['%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']
    )
    
    class Meta:
        model = FruitBatch
        fields = ['batch_number', 'fruit_type', 'product', 'quantity',
                 'arrival_date', 'expected_expiry', 'supplier',
                 'storage_location', 'initial_quality']
        widgets = {
            'batch_number': forms.TextInput(attrs={'class': 'form-control'}),
            'fruit_type': forms.Select(attrs={'class': 'form-control'}),
            'product': forms.Select(attrs={'class': 'form-control'}),
            'quantity': forms.NumberInput(attrs={'class': 'form-control'}),
            'arrival_date': forms.DateTimeInput(
                attrs={'class': 'form-control', 'type': 'datetime-local'},
                format='%Y-%m-%dT%H:%M'
            ),
            'supplier': forms.TextInput(attrs={'class': 'form-control'}),
            'storage_location': forms.Select(attrs={'class': 'form-control'}),
            'initial_quality': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Limit products to those belonging to the vendor
        if self.user and not self.user.is_staff:
            self.fields['product'].queryset = Product.objects.filter(vendor=self.user)
        else:
            self.fields['product'].queryset = Product.objects.all()
    
    def clean_batch_number(self):
        batch_number = self.cleaned_data.get('batch_number')
        if batch_number and FruitBatch.objects.filter(batch_number=batch_number).exclude(pk=self.instance.pk).exists():
            raise ValidationError("A batch with this number already exists.")
        return batch_number
    
    def clean_quantity(self):
        quantity = self.cleaned_data.get('quantity')
        if quantity and quantity < 0:
            raise ValidationError("Quantity cannot be negative.")
        return quantity
    
    def clean_expected_expiry(self):
        expected_expiry = self.cleaned_data.get('expected_expiry')
        arrival_date = self.cleaned_data.get('arrival_date')
        
        if expected_expiry and arrival_date and expected_expiry <= arrival_date:
            raise ValidationError("Expected expiry must be after arrival date.")
        
        return expected_expiry

class FruitQualityReadingForm(forms.ModelForm):
    class Meta:
        model = FruitQualityReading
        fields = ['temperature', 'humidity', 'light_intensity', 'co2_level',
                 'actual_class', 'predicted_class', 'confidence_score',
                 'ethylene_level', 'weight_loss', 'firmness',
                 'model_used', 'model_version', 'notes']
        widgets = {
            'temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'humidity': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'light_intensity': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'co2_level': forms.NumberInput(attrs={'class': 'form-control'}),
            'actual_class': forms.Select(attrs={'class': 'form-control'}),
            'predicted_class': forms.Select(attrs={'class': 'form-control'}),
            'confidence_score': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0', 'max': '1'}),
            'ethylene_level': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'weight_loss': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0', 'max': '100'}),
            'firmness': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'model_used': forms.TextInput(attrs={'class': 'form-control'}),
            'model_version': forms.TextInput(attrs={'class': 'form-control'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial timestamp to now
        if not self.instance.pk:
            self.initial['timestamp'] = timezone.now()

class RealTimeSensorDataForm(forms.ModelForm):
    class Meta:
        model = RealTimeSensorData
        fields = ['product', 'fruit_batch', 'sensor_type', 'value', 'unit',
                 'location', 'predicted_class', 'condition_confidence']
        widgets = {
            'product': forms.Select(attrs={'class': 'form-control'}),
            'fruit_batch': forms.Select(attrs={'class': 'form-control'}),
            'sensor_type': forms.Select(attrs={'class': 'form-control'}),
            'value': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.001'}),
            'unit': forms.TextInput(attrs={'class': 'form-control'}),
            'location': forms.Select(attrs={'class': 'form-control'}),
            'predicted_class': forms.TextInput(attrs={'class': 'form-control'}),
            'condition_confidence': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0', 'max': '1'}),
        }

# ==================== AI & DATASET FORMS ====================

class ProductDatasetForm(forms.ModelForm):
    class Meta:
        model = ProductDataset
        fields = ['name', 'dataset_type', 'description', 'data_file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'dataset_type': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'data_file': forms.FileInput(attrs={'class': 'form-control'}),
        }

class TrainedModelForm(forms.ModelForm):
    class Meta:
        model = TrainedModel
        fields = ['name', 'model_type', 'dataset', 'model_file', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'model_type': forms.Select(attrs={'class': 'form-control'}),
            'dataset': forms.Select(attrs={'class': 'form-control'}),
            'model_file': forms.FileInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class FruitQualityPredictionForm(forms.Form):
    fruit_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Banana, Apple, Mango'
        })
    )
    temperature = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'Temperature in °C'
        })
    )
    humidity = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'Humidity in %'
        })
    )
    light_intensity = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'Light intensity in lux'
        })
    )
    co2_level = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1',
            'placeholder': 'CO₂ level in ppm'
        })
    )
    batch_id = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Optional: Batch ID'
        })
    )

# ==================== ALERT & NOTIFICATION FORMS ====================

class ProductAlertForm(forms.ModelForm):
    class Meta:
        model = ProductAlert
        fields = ['product', 'alert_type', 'severity', 'message', 'detected_by', 'is_resolved']
        widgets = {
            'product': forms.Select(attrs={'class': 'form-control'}),
            'alert_type': forms.Select(attrs={'class': 'form-control'}),
            'severity': forms.Select(attrs={'class': 'form-control'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'detected_by': forms.Select(attrs={'class': 'form-control'}),
            'is_resolved': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class AlertResolutionForm(forms.ModelForm):
    class Meta:
        model = ProductAlert
        fields = ['is_resolved', 'resolved_by']
        widgets = {
            'is_resolved': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

# ==================== STORAGE LOCATION FORMS ====================

class StorageLocationForm(forms.ModelForm):
    class Meta:
        model = StorageLocation
        fields = ['name', 'address', 'latitude', 'longitude', 'capacity', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'latitude': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.000001'}),
            'longitude': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.000001'}),
            'capacity': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

# ==================== SITE CONTENT FORMS ====================

class ContactForm(forms.ModelForm):
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'phone', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your Full Name'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'your.email@example.com'
            }),
            'phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your Phone Number'
            }),
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Subject of your message'
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Your message...',
                'rows': 5
            }),
        }
    
    def clean_phone(self):
        phone = self.cleaned_data.get('phone')
        if phone and not phone.replace(' ', '').replace('-', '').replace('+', '').isdigit():
            raise ValidationError("Please enter a valid phone number.")
        return phone

class NewsletterForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address'
        })
    )

class SiteInfoForm(forms.ModelForm):
    class Meta:
        model = SiteInfo
        fields = ['name', 'tagline', 'description', 'email', 'phone', 'address',
                 'logo', 'favicon', 'facebook_url', 'twitter_url', 'instagram_url',
                 'linkedin_url', 'meta_title', 'meta_description']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'tagline': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'logo': forms.FileInput(attrs={'class': 'form-control'}),
            'favicon': forms.FileInput(attrs={'class': 'form-control'}),
            'facebook_url': forms.URLInput(attrs={'class': 'form-control'}),
            'twitter_url': forms.URLInput(attrs={'class': 'form-control'}),
            'instagram_url': forms.URLInput(attrs={'class': 'form-control'}),
            'linkedin_url': forms.URLInput(attrs={'class': 'form-control'}),
            'meta_title': forms.TextInput(attrs={'class': 'form-control'}),
            'meta_description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
        }

class ServiceForm(forms.ModelForm):
    class Meta:
        model = Service
        fields = ['name', 'slug', 'description', 'icon', 'image', 'display_order', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'slug': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'icon': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'fas fa-icon-name'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'display_order': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class TestimonialForm(forms.ModelForm):
    class Meta:
        model = Testimonial
        fields = ['name', 'position', 'company', 'content', 'image', 'rating', 'is_featured', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'position': forms.TextInput(attrs={'class': 'form-control'}),
            'company': forms.TextInput(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'rating': forms.Select(attrs={'class': 'form-control'}),
            'is_featured': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class FAQForm(forms.ModelForm):
    class Meta:
        model = FAQ
        fields = ['question', 'answer', 'display_order', 'is_active']
        widgets = {
            'question': forms.TextInput(attrs={'class': 'form-control'}),
            'answer': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'display_order': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

# ==================== PAYMENT FORMS ====================

class PaymentForm(forms.ModelForm):
    class Meta:
        model = Payment
        fields = ['order', 'payment_method', 'amount', 'currency', 'status',
                 'mobile_money_phone', 'mobile_money_provider', 'transaction_id']
        widgets = {
            'order': forms.Select(attrs={'class': 'form-control'}),
            'payment_method': forms.Select(attrs={'class': 'form-control'}),
            'amount': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'currency': forms.Select(attrs={'class': 'form-control'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
            'mobile_money_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'mobile_money_provider': forms.TextInput(attrs={'class': 'form-control'}),
            'transaction_id': forms.TextInput(attrs={'class': 'form-control'}),
        }

class PaymentGatewaySettingsForm(forms.ModelForm):
    class Meta:
        model = PaymentGatewaySettings
        fields = ['gateway', 'is_active', 'display_name', 'supported_countries',
                 'supported_currencies', 'api_key', 'api_secret', 'merchant_id',
                 'webhook_secret', 'base_url', 'callback_url', 'environment',
                 'transaction_fee_percent', 'transaction_fee_fixed']
        widgets = {
            'gateway': forms.Select(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'display_name': forms.TextInput(attrs={'class': 'form-control'}),
            'api_key': forms.TextInput(attrs={'class': 'form-control', 'type': 'password'}),
            'api_secret': forms.TextInput(attrs={'class': 'form-control', 'type': 'password'}),
            'merchant_id': forms.TextInput(attrs={'class': 'form-control'}),
            'webhook_secret': forms.TextInput(attrs={'class': 'form-control', 'type': 'password'}),
            'base_url': forms.URLInput(attrs={'class': 'form-control'}),
            'callback_url': forms.URLInput(attrs={'class': 'form-control'}),
            'environment': forms.Select(attrs={'class': 'form-control'}),
            'transaction_fee_percent': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'transaction_fee_fixed': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        }

class CurrencyExchangeRateForm(forms.ModelForm):
    class Meta:
        model = CurrencyExchangeRate
        fields = ['base_currency', 'target_currency', 'exchange_rate']
        widgets = {
            'base_currency': forms.Select(attrs={'class': 'form-control'}),
            'target_currency': forms.Select(attrs={'class': 'form-control'}),
            'exchange_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.000001'}),
        }

# ==================== INVENTORY MANAGEMENT FORMS ====================

class InventoryItemForm(forms.ModelForm):
    """Form for creating/editing inventory items"""
    class Meta:
        model = InventoryItem
        fields = [
            'name', 'sku', 'description', 'category', 'product',
            'quantity', 'unit_price', 'low_stock_threshold', 'reorder_point',
            'item_type', 'status', 'location', 'storage_reference', 'batch_number',
            'expiry_date', 'manufactured_date', 'client', 'quality_rating',
            'condition_notes', 'weight_kg', 'dimensions', 'next_check_date'
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Item Name'}),
            'sku': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'SKU-001'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'category': forms.Select(attrs={'class': 'form-control'}),
            'product': forms.Select(attrs={'class': 'form-control'}),
            'quantity': forms.NumberInput(attrs={'class': 'form-control', 'min': 0}),
            'unit_price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'low_stock_threshold': forms.NumberInput(attrs={'class': 'form-control', 'min': 1}),
            'reorder_point': forms.NumberInput(attrs={'class': 'form-control', 'min': 1}),
            'item_type': forms.Select(attrs={'class': 'form-control'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
            'location': forms.Select(attrs={'class': 'form-control'}),
            'storage_reference': forms.TextInput(attrs={'class': 'form-control'}),
            'batch_number': forms.Select(attrs={'class': 'form-control'}),
            'expiry_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'manufactured_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'client': forms.Select(attrs={'class': 'form-control'}),
            'quality_rating': forms.Select(attrs={'class': 'form-control'}),
            'condition_notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
            'weight_kg': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.001'}),
            'dimensions': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'LxWxH (cm)'}),
            'next_check_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # For non-staff users, limit client field to themselves
        if self.user and not self.user.is_staff:
            self.fields['client'].queryset = CustomUser.objects.filter(id=self.user.id)
            self.fields['client'].initial = self.user
            self.fields['client'].widget = forms.HiddenInput()
    
    def clean_sku(self):
        sku = self.cleaned_data.get('sku')
        if sku and InventoryItem.objects.filter(sku=sku).exclude(pk=self.instance.pk).exists():
            raise ValidationError("An item with this SKU already exists.")
        return sku
    
    def clean_quantity(self):
        quantity = self.cleaned_data.get('quantity')
        if quantity and quantity < 0:
            raise ValidationError("Quantity cannot be negative.")
        return quantity
    
    def clean_unit_price(self):
        unit_price = self.cleaned_data.get('unit_price')
        if unit_price and unit_price < 0:
            raise ValidationError("Unit price cannot be negative.")
        return unit_price

class InventoryCheckForm(forms.Form):
    """Form for inventory checking"""
    sku = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Scan or enter SKU'
        })
    )
    action = forms.ChoiceField(
        choices=[
            ('check_in', 'Check In'),
            ('check_out', 'Check Out'),
            ('transfer', 'Transfer'),
            ('adjust', 'Adjust'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    quantity = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    location = forms.ModelChoiceField(
        queryset=StorageLocation.objects.filter(is_active=True),
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 2,
            'placeholder': 'Additional notes'
        })
    )

class InventoryTransferForm(forms.Form):
    """Form for transferring inventory between locations"""
    item = forms.ModelChoiceField(
        queryset=InventoryItem.objects.filter(status='active'),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    from_location = forms.ModelChoiceField(
        queryset=StorageLocation.objects.filter(is_active=True),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    to_location = forms.ModelChoiceField(
        queryset=StorageLocation.objects.filter(is_active=True),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    quantity = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 2
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        from_location = cleaned_data.get('from_location')
        to_location = cleaned_data.get('to_location')
        
        if from_location and to_location and from_location == to_location:
            raise ValidationError("Source and destination locations must be different.")
        
        return cleaned_data

# ==================== DELIVERY MANAGEMENT FORMS ====================

class DeliveryForm(forms.ModelForm):
    """Form for creating/editing deliveries"""
    class Meta:
        model = Delivery
        fields = [
            'client', 'client_name', 'client_address', 'client_phone', 'client_email',
            'delivery_address', 'delivery_city', 'delivery_state', 'delivery_country',
            'delivery_postal_code', 'special_instructions', 'delivery_type',
            'estimated_delivery', 'scheduled_for', 'delivery_window_start', 
            'delivery_window_end', 'assigned_to', 'driver_name', 'driver_phone',
            'vehicle_number', 'package_count', 'total_weight', 'package_dimensions',
            'insurance_amount', 'delivery_cost', 'delivery_tax'
        ]
        widgets = {
            'client': forms.Select(attrs={'class': 'form-control'}),
            'client_name': forms.TextInput(attrs={'class': 'form-control'}),
            'client_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'client_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'client_email': forms.EmailInput(attrs={'class': 'form-control'}),
            'delivery_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'delivery_city': forms.TextInput(attrs={'class': 'form-control'}),
            'delivery_state': forms.TextInput(attrs={'class': 'form-control'}),
            'delivery_country': forms.TextInput(attrs={'class': 'form-control'}),
            'delivery_postal_code': forms.TextInput(attrs={'class': 'form-control'}),
            'special_instructions': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'delivery_type': forms.Select(attrs={'class': 'form-control'}),
            'estimated_delivery': forms.DateTimeInput(attrs={
                'class': 'form-control',
                'type': 'datetime-local'
            }),
            'scheduled_for': forms.DateTimeInput(attrs={
                'class': 'form-control',
                'type': 'datetime-local'
            }),
            'delivery_window_start': forms.TimeInput(attrs={
                'class': 'form-control',
                'type': 'time'
            }),
            'delivery_window_end': forms.TimeInput(attrs={
                'class': 'form-control',
                'type': 'time'
            }),
            'assigned_to': forms.Select(attrs={'class': 'form-control'}),
            'driver_name': forms.TextInput(attrs={'class': 'form-control'}),
            'driver_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'vehicle_number': forms.TextInput(attrs={'class': 'form-control'}),
            'package_count': forms.NumberInput(attrs={'class': 'form-control'}),
            'total_weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'package_dimensions': forms.TextInput(attrs={'class': 'form-control'}),
            'insurance_amount': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'delivery_cost': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'delivery_tax': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        }
    
    def clean_estimated_delivery(self):
        estimated_delivery = self.cleaned_data.get('estimated_delivery')
        if estimated_delivery and estimated_delivery < timezone.now():
            raise ValidationError("Estimated delivery date cannot be in the past.")
        return estimated_delivery

class DeliveryStatusUpdateForm(forms.Form):
    """Form for updating delivery status"""
    status = forms.ChoiceField(
        choices=Delivery.STATUS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Add status update notes...'
        })
    )
    location = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Current location'
        })
    )
    latitude = forms.DecimalField(
        required=False,
        max_digits=9,
        decimal_places=6,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    longitude = forms.DecimalField(
        required=False,
        max_digits=9,
        decimal_places=6,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

class DeliveryProofForm(forms.Form):
    """Form for proof of delivery"""
    recipient_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    recipient_phone = forms.CharField(
        max_length=20,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    recipient_signature = forms.CharField(
        required=False,
        widget=forms.HiddenInput()
    )
    delivery_notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Delivery notes...'
        })
    )
    proof_photo = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

# ==================== USER ROLE & PERMISSION FORMS ====================

class UserRoleForm(forms.ModelForm):
    """Form for assigning user roles"""
    class Meta:
        model = UserRole
        fields = ['user', 'role', 'permissions']
        widgets = {
            'user': forms.Select(attrs={'class': 'form-control'}),
            'role': forms.Select(attrs={'class': 'form-control form-select'}),
            'permissions': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter permissions as JSON (e.g., {"allowed": ["view_inventory", "edit_deliveries"], "denied": ["delete_product"]})'
            }),
        }
    
    def clean_permissions(self):
        permissions = self.cleaned_data.get('permissions')
        if permissions:
            try:
                if isinstance(permissions, str):
                    permissions_data = json.loads(permissions)
                else:
                    permissions_data = permissions

                if not isinstance(permissions_data, dict) or \
                   'allowed' not in permissions_data or \
                   'denied' not in permissions_data:
                    raise ValidationError("Permissions must be a dictionary with 'allowed' and 'denied' keys.")
                
                if not isinstance(permissions_data['allowed'], list) or \
                   not isinstance(permissions_data['denied'], list):
                    raise ValidationError("Permissions 'allowed' and 'denied' must be lists.")
                
                return json.dumps(permissions_data) if isinstance(permissions, str) else permissions_data
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON format for permissions.")
            except Exception as e:
                raise ValidationError(f"Error parsing permissions: {e}")
        return {}

class RoleAssignmentForm(forms.Form):
    """Form for bulk role assignment"""
    users = forms.ModelMultipleChoiceField(
        queryset=CustomUser.objects.all(),
        widget=forms.SelectMultiple(attrs={'class': 'form-control'})
    )
    role = forms.ChoiceField(
        choices=UserRole.ROLE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

# ==================== AI & MODEL TRAINING FORMS ====================

class ModelTrainingForm(forms.Form):
    """Form for training AI models"""
    dataset = forms.ModelChoiceField(
        queryset=ProductDataset.objects.filter(is_active=True),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    model_type = forms.ChoiceField(
        choices=[
            ('random_forest', 'Random Forest'),
            ('xgboost', 'XGBoost'),
            ('svm', 'Support Vector Machine'),
            ('neural_network', 'Neural Network'),
            ('gradient_boosting', 'Gradient Boosting'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    target_column = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Column name to predict'
        })
    )
    test_size = forms.FloatField(
        min_value=0.1,
        max_value=0.5,
        initial=0.2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05'
        })
    )
    random_state = forms.IntegerField(
        min_value=0,
        initial=42,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

class DatasetUploadForm(forms.Form):
    """Form for uploading datasets"""
    name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    dataset_type = forms.ChoiceField(
        choices=ProductDataset.DATASET_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3
        })
    )
    data_file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

# ==================== QUALITY READING & SENSOR FORMS ====================

class BatchQualityReadingForm(forms.Form):
    """Form for batch quality readings"""
    batch = forms.ModelChoiceField(
        queryset=FruitBatch.objects.filter(status='active'),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    temperature = forms.DecimalField(
        max_digits=5,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    humidity = forms.DecimalField(
        max_digits=5,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    light_intensity = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    co2_level = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 2
        })
    )

class SensorConfigurationForm(forms.Form):
    """Form for sensor configuration"""
    sensor_type = forms.ChoiceField(
        choices=RealTimeSensorData.SENSOR_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    location = forms.ModelChoiceField(
        queryset=StorageLocation.objects.filter(is_active=True),
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    reading_interval = forms.ChoiceField(
        choices=[
            ('5min', 'Every 5 minutes'),
            ('15min', 'Every 15 minutes'),
            ('30min', 'Every 30 minutes'),
            ('1hour', 'Every hour'),
            ('4hour', 'Every 4 hours'),
            ('daily', 'Daily'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    threshold_min = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    threshold_max = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

# ==================== REPORT & ANALYTICS FORMS ====================

class ReportFilterForm(forms.Form):
    """Form for filtering reports"""
    REPORT_TYPES = [
        ('inventory', 'Inventory Report'),
        ('deliveries', 'Delivery Report'),
        ('sales', 'Sales Report'),
        ('quality', 'Quality Report'),
        ('revenue', 'Revenue Report'),
    ]
    
    report_type = forms.ChoiceField(
        choices=REPORT_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    group_by = forms.ChoiceField(
        choices=[
            ('daily', 'Daily'),
            ('weekly', 'Weekly'),
            ('monthly', 'Monthly'),
            ('quarterly', 'Quarterly'),
            ('yearly', 'Yearly'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class AnalyticsDashboardForm(forms.Form):
    """Form for customizing analytics dashboard"""
    metrics = forms.MultipleChoiceField(
        choices=[
            ('total_items', 'Total Items'),
            ('inventory_value', 'Inventory Value'),
            ('low_stock_count', 'Low Stock Items'),
            ('delivery_success_rate', 'Delivery Success Rate'),
            ('quality_trend', 'Quality Trend'),
            ('revenue_trend', 'Revenue Trend'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    time_period = forms.ChoiceField(
        choices=[
            ('7d', 'Last 7 days'),
            ('30d', 'Last 30 days'),
            ('90d', 'Last 90 days'),
            ('1y', 'Last year'),
            ('custom', 'Custom range'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    refresh_interval = forms.ChoiceField(
        choices=[
            ('5min', '5 minutes'),
            ('15min', '15 minutes'),
            ('30min', '30 minutes'),
            ('1hour', '1 hour'),
            ('manual', 'Manual refresh'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

# ==================== EMAIL & NOTIFICATION FORMS ====================

class NotificationSettingsForm(forms.ModelForm):
    """Form for notification settings"""
    class Meta:
        model = NotificationSettings
        fields = [
            'email_notifications', 'push_notifications', 'sms_notifications',
            'order_updates', 'delivery_updates', 'inventory_alerts',
            'quality_alerts', 'system_alerts', 'promotions'
        ]
        widgets = {
            'email_notifications': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'push_notifications': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'sms_notifications': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'order_updates': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'delivery_updates': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'inventory_alerts': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'quality_alerts': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'system_alerts': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'promotions': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class EmailTemplateForm(forms.Form):
    """Form for email templates"""
    template_type = forms.ChoiceField(
        choices=[
            ('welcome', 'Welcome Email'),
            ('order_confirmation', 'Order Confirmation'),
            ('delivery_update', 'Delivery Update'),
            ('quality_alert', 'Quality Alert'),
            ('inventory_alert', 'Inventory Alert'),
            ('promotional', 'Promotional'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    subject = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    body = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 8,
            'placeholder': 'Use {{variable}} for dynamic content'
        })
    )
    is_active = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# ==================== SYSTEM CONFIGURATION FORMS ====================

class SystemSettingsForm(forms.Form):
    """Form for system settings"""
    site_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    timezone = forms.ChoiceField(
        choices=[
            ('Africa/Dar_es_Salaam', 'Dar es Salaam (GMT+3)'),
            ('UTC', 'UTC'),
            ('America/New_York', 'New York (GMT-5)'),
            ('Europe/London', 'London (GMT+0)'),
            ('Asia/Tokyo', 'Tokyo (GMT+9)'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    currency = forms.ChoiceField(
        choices=Payment.CURRENCIES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    language = forms.ChoiceField(
        choices=[
            ('en', 'English'),
            ('sw', 'Swahili'),
            ('fr', 'French'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    maintenance_mode = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    debug_mode = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# ==================== QUICK ACTION FORMS ====================

class QuickInventoryActionForm(forms.Form):
    """Form for quick inventory actions"""
    action = forms.ChoiceField(
        choices=[
            ('check_in', 'Quick Check In'),
            ('check_out', 'Quick Check Out'),
            ('adjust', 'Quick Adjust'),
        ],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    sku = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter SKU'
        })
    )
    quantity = forms.IntegerField(
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    location = forms.ModelChoiceField(
        queryset=StorageLocation.objects.filter(is_active=True),
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class QuickDeliveryUpdateForm(forms.Form):
    """Form for quick delivery updates"""
    delivery_number = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Delivery Number'
        })
    )
    status = forms.ChoiceField(
        choices=[
            ('in_transit', 'In Transit'),
            ('out_for_delivery', 'Out for Delivery'),
            ('delivered', 'Delivered'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    location = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Current Location'
        })
    )

# ==================== BULK ACTION FORMS ====================

class BulkProductActionForm(forms.Form):
    ACTION_CHOICES = [
        ('activate', 'Activate Selected'),
        ('draft', 'Move to Draft'),
        ('delete', 'Delete Selected'),
        ('feature', 'Mark as Featured'),
        ('unfeature', 'Remove Featured Status'),
    ]
    
    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    product_ids = forms.CharField(
        widget=forms.HiddenInput()
    )
    
    def clean_product_ids(self):
        product_ids = self.cleaned_data.get('product_ids', '')
        try:
            ids = [int(id.strip()) for id in product_ids.split(',') if id.strip()]
            return ids
        except ValueError:
            raise ValidationError("Invalid product IDs format.")

# ==================== CLIENT REQUEST FORM ====================

class ClientRequestForm(forms.ModelForm):
    inventory_items = forms.ModelMultipleChoiceField(
        queryset=InventoryItem.objects.none(),
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = ClientRequest
        fields = [
            'request_type', 'title', 'description', 'quantity', 
            'urgency', 'preferred_delivery_date', 'notes', 'inventory_items'
        ]
        widgets = {
            'request_type': forms.Select(attrs={'class': 'form-control'}),
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Request Title'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe your request in detail'
            }),
            'quantity': forms.NumberInput(attrs={'class': 'form-control', 'min': 1}),
            'urgency': forms.Select(attrs={'class': 'form-control'}),
            'preferred_delivery_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date'
            }),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Additional notes or instructions'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if self.user:
            self.fields['inventory_items'].queryset = InventoryItem.objects.filter(
                client=self.user
            )
    
    def clean_preferred_delivery_date(self):
        preferred_delivery_date = self.cleaned_data.get('preferred_delivery_date')
        if preferred_delivery_date and preferred_delivery_date < timezone.now().date():
            raise ValidationError("Preferred delivery date cannot be in the past.")
        return preferred_delivery_date
    
    def clean_quantity(self):
        quantity = self.cleaned_data.get('quantity')
        if quantity and quantity <= 0:
            raise ValidationError("Quantity must be greater than 0.")
        return quantity

# ==================== EXPORT FORMS ====================

class ExportForm(forms.Form):
    EXPORT_TYPES = [
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('pdf', 'PDF'),
        ('json', 'JSON'),
    ]
    
    export_type = forms.ChoiceField(
        choices=EXPORT_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    include_images = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    date_range = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    start_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    end_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )

# ==================== IMPORT FORMS ====================

class ImportForm(forms.Form):
    IMPORT_TYPES = [
        ('products', 'Products'),
        ('inventory', 'Inventory Items'),
        ('customers', 'Customers'),
        ('orders', 'Orders'),
        ('deliveries', 'Deliveries'),
    ]
    
    import_type = forms.ChoiceField(
        choices=IMPORT_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    import_file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    update_existing = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    skip_errors = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# ==================== BACKUP & RESTORE FORMS ====================

class BackupForm(forms.Form):
    BACKUP_TYPES = [
        ('full', 'Full Backup'),
        ('database', 'Database Only'),
        ('media', 'Media Files Only'),
        ('settings', 'Settings Only'),
    ]
    
    backup_type = forms.ChoiceField(
        choices=BACKUP_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    include_logs = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    compress = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    encryption_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Optional encryption key'
        })
    )

class RestoreForm(forms.Form):
    backup_file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    decryption_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter decryption key if backup is encrypted'
        })
    )
    verify_data = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# ==================== PASSWORD & SECURITY FORMS ====================

class ChangePasswordForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

class TwoFactorSetupForm(forms.ModelForm):
    """Form for 2FA settings"""
    enable_2fa = forms.BooleanField(
        label="Enable Two-Factor Authentication",
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    class Meta:
        model = TwoFactorSettings
        fields = ['is_enabled', 'method', 'phone_number']
        widgets = {
            'method': forms.Select(attrs={'class': 'form-control'}),
            'phone_number': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Phone number for SMS verification'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance:
            self.initial['enable_2fa'] = self.instance.is_enabled
        self.fields['is_enabled'].widget = forms.HiddenInput()

    def clean(self):
        cleaned_data = super().clean()
        enable_2fa = cleaned_data.get('enable_2fa')
        method = cleaned_data.get('method')
        phone_number = cleaned_data.get('phone_number')

        if enable_2fa and method == 'sms' and not phone_number:
            self.add_error('phone_number', "Phone number is required for SMS 2FA.")
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.is_enabled = self.cleaned_data.get('enable_2fa')
        if commit:
            instance.save()
        return instance

class APIKeyForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'API Key Name'
        })
    )
    permissions = forms.MultipleChoiceField(
        choices=[
            ('read', 'Read Only'),
            ('write', 'Write Access'),
            ('delete', 'Delete Access'),
            ('admin', 'Admin Access'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    expires_in = forms.ChoiceField(
        choices=[
            ('7', '7 days'),
            ('30', '30 days'),
            ('90', '90 days'),
            ('365', '1 year'),
            ('never', 'Never expires'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

# ==================== MISC UTILITY FORMS ====================

class SearchForm(forms.Form):
    """Generic search form"""
    query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search...'
        })
    )
    search_type = forms.ChoiceField(
        choices=[
            ('all', 'All'),
            ('products', 'Products'),
            ('inventory', 'Inventory'),
            ('customers', 'Customers'),
            ('orders', 'Orders'),
            ('deliveries', 'Deliveries'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class DateRangeForm(forms.Form):
    """Form for selecting date range"""
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get('start_date')
        end_date = cleaned_data.get('end_date')
        
        if start_date and end_date and start_date > end_date:
            raise ValidationError("Start date cannot be after end date.")
        
        return cleaned_data

class UploadFileForm(forms.Form):
    """Generic file upload form"""
    file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 2,
            'placeholder': 'File description'
        })
    )
    is_public = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# ==================== DASHBOARD WIDGET FORMS ====================

class DashboardWidgetForm(forms.Form):
    """Form for customizing dashboard widgets"""
    widgets = forms.MultipleChoiceField(
        choices=[
            ('stats', 'Statistics Overview'),
            ('recent_orders', 'Recent Orders'),
            ('inventory_alerts', 'Inventory Alerts'),
            ('quality_monitor', 'Quality Monitor'),
            ('revenue_chart', 'Revenue Chart'),
            ('delivery_status', 'Delivery Status'),
            ('top_products', 'Top Products'),
            ('activity_feed', 'Activity Feed'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    layout = forms.ChoiceField(
        choices=[
            ('grid', 'Grid Layout'),
            ('list', 'List Layout'),
            ('compact', 'Compact Layout'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    refresh_rate = forms.IntegerField(
        min_value=30,
        max_value=3600,
        initial=300,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Refresh rate in seconds'
        })
    )

# ==================== API ENDPOINT FORMS ====================

class APIEndpointForm(forms.Form):
    """Form for API endpoint configuration"""
    endpoint_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Endpoint Name'
        })
    )
    endpoint_url = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '/api/endpoint/'
        })
    )
    http_method = forms.ChoiceField(
        choices=[
            ('GET', 'GET'),
            ('POST', 'POST'),
            ('PUT', 'PUT'),
            ('DELETE', 'DELETE'),
            ('PATCH', 'PATCH'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    authentication = forms.ChoiceField(
        choices=[
            ('none', 'No Authentication'),
            ('token', 'Token Authentication'),
            ('basic', 'Basic Authentication'),
            ('oauth', 'OAuth 2.0'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    rate_limit = forms.IntegerField(
        min_value=1,
        max_value=10000,
        initial=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Requests per hour'
        })
    )

# ==================== WEBHOOK FORMS ====================

class WebhookForm(forms.Form):
    """Form for webhook configuration"""
    webhook_url = forms.URLField(
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://example.com/webhook'
        })
    )
    events = forms.MultipleChoiceField(
        choices=[
            ('order_created', 'Order Created'),
            ('order_updated', 'Order Updated'),
            ('delivery_created', 'Delivery Created'),
            ('delivery_updated', 'Delivery Updated'),
            ('inventory_low', 'Inventory Low Stock'),
            ('quality_alert', 'Quality Alert'),
            ('payment_received', 'Payment Received'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'})
    )
    secret_key = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Webhook secret key'
        })
    )
    is_active = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# ==================== CUSTOM FIELD FORMS ====================

class CustomFieldForm(forms.Form):
    """Form for creating custom fields"""
    FIELD_TYPES = [
        ('text', 'Text'),
        ('number', 'Number'),
        ('date', 'Date'),
        ('boolean', 'Yes/No'),
        ('select', 'Dropdown'),
        ('multiselect', 'Multi-select'),
        ('file', 'File Upload'),
        ('image', 'Image Upload'),
    ]
    
    field_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Field Name'
        })
    )
    field_type = forms.ChoiceField(
        choices=FIELD_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    model = forms.ChoiceField(
        choices=[
            ('product', 'Product'),
            ('inventory', 'Inventory Item'),
            ('customer', 'Customer'),
            ('order', 'Order'),
            ('delivery', 'Delivery'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    is_required = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    default_value = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Default Value'
        })
    )
    options = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Options for select fields (comma separated)'
        })
    )

# ==================== LOG VIEWER FORMS ====================

class LogFilterForm(forms.Form):
    """Form for filtering logs"""
    LOG_LEVELS = [
        ('DEBUG', 'DEBUG'),
        ('INFO', 'INFO'),
        ('WARNING', 'WARNING'),
        ('ERROR', 'ERROR'),
        ('CRITICAL', 'CRITICAL'),
    ]
    
    log_level = forms.ChoiceField(
        choices=[('', 'All Levels')] + LOG_LEVELS,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    logger_name = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Logger Name'
        })
    )
    message = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search in messages'
        })
    )
    start_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    end_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )

# ==================== BATCH OPERATION FORMS ====================

class BatchOperationForm(forms.Form):
    """Form for batch operations"""
    OPERATION_TYPES = [
        ('update_status', 'Update Status'),
        ('assign_category', 'Assign Category'),
        ('update_price', 'Update Price'),
        ('adjust_quantity', 'Adjust Quantity'),
        ('delete', 'Delete'),
        ('export', 'Export'),
    ]
    
    operation_type = forms.ChoiceField(
        choices=OPERATION_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    model_type = forms.ChoiceField(
        choices=[
            ('product', 'Products'),
            ('inventory', 'Inventory Items'),
            ('order', 'Orders'),
            ('delivery', 'Deliveries'),
            ('customer', 'Customers'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    filter_criteria = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Filter criteria (JSON format)'
        })
    )
    action_value = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Value for the operation'
        })
    )

# ==================== TEMPLATE FORMS ====================

class TemplateForm(forms.Form):
    """Form for managing templates"""
    TEMPLATE_TYPES = [
        ('email', 'Email Template'),
        ('sms', 'SMS Template'),
        ('report', 'Report Template'),
        ('invoice', 'Invoice Template'),
        ('receipt', 'Receipt Template'),
    ]
    
    template_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Template Name'
        })
    )
    template_type = forms.ChoiceField(
        choices=TEMPLATE_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    content = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 10,
            'placeholder': 'Template content with {{variables}}'
        })
    )
    variables = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Available variables (one per line)'
        })
    )

# ==================== FINAL EXPORT LIST ====================

__all__ = [
    # Authentication Forms
    'LoginForm', 'CustomUserCreationForm', 'VendorRegistrationForm', 'CustomerRegistrationForm',
    
    # User Profile Forms
    'UserProfileForm', 'VendorProfileForm',
    
    # Address Form
    'AddressForm',
    
    # Product Forms
    'ProductForm', 'ProductImageForm', 'ProductImageInlineForm', 'ProductReviewForm', 'ClientProductCreationForm',
    
    # Category Forms
    'ProductCategoryForm',
    
    # Search & Filter Forms
    'ProductSearchForm', 'ProductFilterForm',
    
    # Cart & Order Forms
    'CartItemForm', 'CheckoutForm',
    
    # Fruit Quality Forms
    'FruitTypeForm', 'FruitBatchForm', 'FruitQualityReadingForm', 'RealTimeSensorDataForm',
    'FruitQualityPredictionForm',
    
    # AI & Dataset Forms
    'ProductDatasetForm', 'TrainedModelForm', 'ModelTrainingForm', 'DatasetUploadForm',
    
    # Alert & Notification Forms
    'ProductAlertForm', 'AlertResolutionForm', 'NotificationSettingsForm', 'EmailTemplateForm',
    
    # Storage Forms
    'StorageLocationForm',
    
    # Site Content Forms
    'ContactForm', 'NewsletterForm', 'SiteInfoForm', 'ServiceForm', 'TestimonialForm', 'FAQForm',
    
    # Payment Forms
    'PaymentForm', 'PaymentGatewaySettingsForm', 'CurrencyExchangeRateForm',
    
    # Inventory Forms
    'InventoryItemForm', 'InventoryCheckForm', 'InventoryTransferForm',
    
    # Delivery Forms
    'DeliveryForm', 'DeliveryStatusUpdateForm', 'DeliveryProofForm',
    
    # User Role Forms
    'UserRoleForm', 'RoleAssignmentForm',
    
    # Quality & Sensor Forms
    'BatchQualityReadingForm', 'SensorConfigurationForm',
    
    # Report & Analytics Forms
    'ReportFilterForm', 'AnalyticsDashboardForm',
    
    # System Forms
    'SystemSettingsForm',
    
    # Quick Action Forms
    'QuickInventoryActionForm', 'QuickDeliveryUpdateForm',
    
    # Bulk Action Forms
    'BulkProductActionForm',
    
    # Client Request Forms
    'ClientRequestForm',
    
    # Export/Import Forms
    'ExportForm', 'ImportForm',
    
    # Backup/Restore Forms
    'BackupForm', 'RestoreForm',
    
    # Security Forms
    'ChangePasswordForm', 'TwoFactorSetupForm', 'APIKeyForm',
    
    # Utility Forms
    'SearchForm', 'DateRangeForm', 'UploadFileForm',
    
    # Dashboard Forms
    'DashboardWidgetForm',
    
    # API Forms
    'APIEndpointForm', 'WebhookForm',
    
    # Custom Field Forms
    'CustomFieldForm',
    
    # Log Forms
    'LogFilterForm',
    
    # Batch Operation Forms
    'BatchOperationForm',
    
    # Template Forms
    'TemplateForm',
]