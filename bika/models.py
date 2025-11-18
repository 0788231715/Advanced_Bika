from django.db import models
from django.urls import reverse
from django.utils import timezone

class SiteInfo(models.Model):
    """Store site-wide information"""
    name = models.CharField(max_length=200, default="Bika")
    tagline = models.CharField(max_length=300, blank=True)
    description = models.TextField(blank=True)
    email = models.EmailField(default="contact@bika.com")
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    logo = models.ImageField(upload_to='site/logo/', blank=True)
    favicon = models.ImageField(upload_to='site/favicon/', blank=True)
    
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
            # Update the existing instance
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
    image = models.ImageField(upload_to='services/', blank=True)
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
    image = models.ImageField(upload_to='testimonials/', blank=True)
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