from django.urls import path
from . import views

app_name = 'bika'

urlpatterns = [
    # Main pages
    path('', views.HomeView.as_view(), name='home'),
    path('about/', views.about_view, name='about'),
    path('services/', views.services_view, name='services'),
    path('services/<slug:slug>/', views.ServiceDetailView.as_view(), name='service_detail'),
    path('contact/', views.contact_view, name='contact'),
    path('faq/', views.faq_view, name='faq'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    # AJAX/API endpoints
    
    path('newsletter/subscribe/', views.newsletter_subscribe, name='newsletter_subscribe'),
]