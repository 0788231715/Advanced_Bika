from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import handler404, handler500

handler404 = 'bika.views.custom_404'
handler500 = 'bika.views.custom_500'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bika.urls')),
    path('accounts/', include('django.contrib.auth.urls')),
    
]

# Custom admin site header
admin.site.site_header = "Bika Administration"
admin.site.site_title = "Bika Admin Portal"
admin.site.index_title = "Welcome to Bika Admin Portal"

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Error handlers
handler404 = 'bika.views.handler404'
handler500 = 'bika.views.handler500'