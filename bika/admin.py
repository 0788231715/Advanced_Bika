from django.contrib import admin
from .models import SiteInfo, Service, Testimonial, ContactMessage, FAQ

@admin.register(SiteInfo)
class SiteInfoAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'updated_at']
    readonly_fields = ['updated_at']
    
    def has_add_permission(self, request):
        # Allow only one instance
        if self.model.objects.count() >= 1:
            return False
        return super().has_add_permission(request)

@admin.register(Service)
class ServiceAdmin(admin.ModelAdmin):
    list_display = ['name', 'display_order', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    list_editable = ['display_order', 'is_active']

@admin.register(Testimonial)
class TestimonialAdmin(admin.ModelAdmin):
    list_display = ['name', 'company', 'rating', 'is_featured', 'is_active', 'created_at']
    list_filter = ['is_featured', 'is_active', 'rating', 'created_at']
    search_fields = ['name', 'company', 'content']
    list_editable = ['is_featured', 'is_active']

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'status', 'submitted_at']
    list_filter = ['status', 'submitted_at']
    search_fields = ['name', 'email', 'subject', 'message']
    readonly_fields = ['submitted_at', 'ip_address']
    actions = ['mark_as_replied', 'mark_as_read']
    
    def mark_as_replied(self, request, queryset):
        for message in queryset:
            message.mark_as_replied()
        self.message_user(request, f"{queryset.count()} messages marked as replied.")
    mark_as_replied.short_description = "Mark selected messages as replied"
    
    def mark_as_read(self, request, queryset):
        queryset.update(status='read')
        self.message_user(request, f"{queryset.count()} messages marked as read.")
    mark_as_read.short_description = "Mark selected messages as read"

@admin.register(FAQ)
class FAQAdmin(admin.ModelAdmin):
    list_display = ['question', 'display_order', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['question', 'answer']
    list_editable = ['display_order', 'is_active']