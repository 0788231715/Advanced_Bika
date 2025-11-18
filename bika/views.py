
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count
import django
import sys
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.views.generic import ListView, DetailView, TemplateView

from .models import SiteInfo, Service, Testimonial, ContactMessage, FAQ
from .forms import ContactForm, NewsletterForm

class HomeView(TemplateView):
    template_name = 'bika/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['featured_services'] = Service.objects.filter(is_active=True)[:6]
        context['featured_testimonials'] = Testimonial.objects.filter(
            is_active=True, 
            is_featured=True
        )[:3]
        context['faqs'] = FAQ.objects.filter(is_active=True)[:5]
        return context

def about_view(request):
    services = Service.objects.filter(is_active=True)
    testimonials = Testimonial.objects.filter(is_active=True)[:4]
    
    context = {
        'services': services,
        'testimonials': testimonials,
    }
    return render(request, 'bika/pages/about.html', context)

def services_view(request):
    services = Service.objects.filter(is_active=True)
    return render(request, 'bika/pages/services.html', {'services': services})

class ServiceDetailView(DetailView):
    model = Service
    template_name = 'bika/pages/service_detail.html'
    context_object_name = 'service'
    slug_field = 'slug'
    slug_url_kwarg = 'slug'

def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            contact_message = form.save(commit=False)
            
            # Get client IP address
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                contact_message.ip_address = x_forwarded_for.split(',')[0]
            else:
                contact_message.ip_address = request.META.get('REMOTE_ADDR')
            
            contact_message.save()
            
            # Send email notification (optional)
            try:
                send_mail(
                    f'New Contact Message: {contact_message.subject}',
                    f'''
                    Name: {contact_message.name}
                    Email: {contact_message.email}
                    Phone: {contact_message.phone}
                    
                    Message:
                    {contact_message.message}
                    ''',
                    settings.DEFAULT_FROM_EMAIL,
                    [settings.DEFAULT_FROM_EMAIL],
                    fail_silently=True,
                )
            except Exception as e:
                # Log error but don't show to user
                pass
            
            messages.success(
                request, 
                'Thank you for your message! We will get back to you soon.'
            )
            return redirect('bika:contact')
    else:
        form = ContactForm()
    
    return render(request, 'bika/pages/contact.html', {'form': form})

def faq_view(request):
    faqs = FAQ.objects.filter(is_active=True)
    return render(request, 'bika/pages/faq.html', {'faqs': faqs})

def newsletter_subscribe(request):
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        form = NewsletterForm(request.POST)
        if form.is_valid():
            # Here you would typically save to database
            # For now, we'll just return success
            email = form.cleaned_data['email']
            return JsonResponse({
                'success': True,
                'message': 'Thank you for subscribing to our newsletter!'
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Please enter a valid email address.'
            })
    return JsonResponse({'success': False, 'message': 'Invalid request'})

def handler404(request, exception):
    return render(request, 'bika/pages/404.html', status=404)

def handler500(request):
    return render(request, 'bika/pages/500.html', status=500)


def custom_404(request, exception):
    return render(request, 'bika/pages/404.html', status=404)

def custom_500(request):
    return render(request, 'bika/pages/500.html', status=500)

# Optional: Test view to trigger 500 error (remove in production)
def test_500(request):
    # This will trigger a 500 error for testing
    raise Exception("This is a test 500 error")


def admin_dashboard(request):
    """Custom admin dashboard"""
    if not request.user.is_staff:
        return redirect('admin:login')
    
    # Get current date and time
    now = timezone.now()
    last_week = now - timedelta(days=7)
    
    # Statistics
    total_services = Service.objects.count()
    total_testimonials = Testimonial.objects.count()
    total_messages = ContactMessage.objects.count()
    new_messages = ContactMessage.objects.filter(status='new').count()
    
    # Additional stats
    active_services_count = Service.objects.filter(is_active=True).count()
    featured_testimonials_count = Testimonial.objects.filter(is_featured=True, is_active=True).count()
    active_faqs_count = FAQ.objects.filter(is_active=True).count()
    
    # Recent activity
    recent_messages = ContactMessage.objects.all().order_by('-submitted_at')[:5]
    
    # System information
    import django
    import sys
    from django.conf import settings
    
    context = {
        # Basic stats
        'total_services': total_services,
        'total_testimonials': total_testimonials,
        'total_messages': total_messages,
        'new_messages': new_messages,
        
        # Additional stats
        'active_services_count': active_services_count,
        'featured_testimonials_count': featured_testimonials_count,
        'active_faqs_count': active_faqs_count,
        
        # Recent activity
        'recent_messages': recent_messages,
        
        # System info
        'django_version': django.get_version(),
        'python_version': sys.version.split()[0],
        'debug': settings.DEBUG,
    }
    
    return render(request, 'admin/dashboard.html', context)