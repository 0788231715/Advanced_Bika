from django.apps import AppConfig

from django.apps import AppConfig

class BikaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bika'
    verbose_name = 'Bika Marketplace'
    
    def ready(self):
        import bika.signals  # if you have signals
class BikaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bika'
