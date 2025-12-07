# bika/forms.py - CORRECTED IMPORT SECTION

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone

# Import models - CORRECTED
from .models import (
    CustomUser, SiteInfo, Service, ProductCategory, Product, 
    ProductImage, ProductReview, Wishlist, Cart, Order, OrderItem, 
    Payment, ContactMessage, FAQ, StorageLocation, FruitType, 
    FruitBatch, FruitQualityReading, RealTimeSensorData, 
    ProductAlert, Notification, ProductDataset, TrainedModel,
    PaymentGatewaySettings, CurrencyExchangeRate, Testimonial
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

# Temporarily comment out this form
# class PaymentGatewaySettingsForm(forms.ModelForm):
#     class Meta:
#         model = PaymentGatewaySettings
#         fields = ['gateway', 'is_active', 'display_name', 'supported_countries',
#                  'supported_currencies', 'api_key', 'api_secret', 'merchant_id',
#                  'webhook_secret', 'base_url', 'callback_url', 'environment',
#                  'transaction_fee_percent', 'transaction_fee_fixed']
#         widgets = {
#             'gateway': forms.Select(attrs={'class': 'form-control'}),
#             'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
#             'display_name': forms.TextInput(attrs={'class': 'form-control'}),
#             'api_key': forms.TextInput(attrs={'class': 'form-control', 'type': 'password'}),
#             'api_secret': forms.TextInput(attrs={'class': 'form-control', 'type': 'password'}),
#             'merchant_id': forms.TextInput(attrs={'class': 'form-control'}),
#             'webhook_secret': forms.TextInput(attrs={'class': 'form-control', 'type': 'password'}),
#             'base_url': forms.URLInput(attrs={'class': 'form-control'}),
#             'callback_url': forms.URLInput(attrs={'class': 'form-control'}),
#             'environment': forms.Select(attrs={'class': 'form-control'}),
#             'transaction_fee_percent': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
#             'transaction_fee_fixed': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
#         }
class CurrencyExchangeRateForm(forms.ModelForm):
    class Meta:
        model = CurrencyExchangeRate
        fields = ['base_currency', 'target_currency', 'exchange_rate']
        widgets = {
            'base_currency': forms.Select(attrs={'class': 'form-control'}),
            'target_currency': forms.Select(attrs={'class': 'form-control'}),
            'exchange_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.000001'}),
        }

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