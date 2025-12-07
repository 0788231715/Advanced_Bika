# bika/apps.py
from django.apps import AppConfig

class BikaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bika'
    verbose_name = 'Bika Marketplace'
    
    def ready(self):
        """Initialize the Bika application"""
        # Import signals if you have them
        try:
            import bika.signals  # noqa: F401
            print("Bika: Signals imported successfully")
        except ImportError:
            print("Bika: No signals module found")
        
        # Initialize any startup tasks here
        self.initialize_default_data()
    
    def initialize_default_data(self):
        """Initialize default data for the application"""
        from django.db.utils import OperationalError, ProgrammingError
        
        try:
            # Check if SiteInfo exists, create default if not
            from .models import SiteInfo
            if not SiteInfo.objects.exists():
                SiteInfo.objects.create(
                    name="Bika",
                    tagline="AI-Powered Fruit Quality Monitoring & E-commerce Platform",
                    description="Your Success Is Our Business - Bika provides exceptional services to help your business grow.",
                    email="contact@bika.com",
                    phone="+255 123 456 789",
                    address="Dar es Salaam, Tanzania",
                )
                print("Bika: Default SiteInfo created")
            
            # Check if default fruit types exist
            from .models import FruitType
            default_fruits = [
                {
                    'name': 'Banana',
                    'scientific_name': 'Musa spp.',
                    'optimal_temp_min': 13.0,
                    'optimal_temp_max': 15.0,
                    'optimal_humidity_min': 90.0,
                    'optimal_humidity_max': 95.0,
                    'optimal_light_max': 100,
                    'optimal_co2_max': 400,
                    'shelf_life_days': 7,
                    'ethylene_sensitive': False,
                    'chilling_sensitive': True,
                },
                {
                    'name': 'Apple',
                    'scientific_name': 'Malus domestica',
                    'optimal_temp_min': 0.0,
                    'optimal_temp_max': 4.0,
                    'optimal_humidity_min': 90.0,
                    'optimal_humidity_max': 95.0,
                    'optimal_light_max': 100,
                    'optimal_co2_max': 400,
                    'shelf_life_days': 30,
                    'ethylene_sensitive': True,
                    'chilling_sensitive': False,
                },
                {
                    'name': 'Mango',
                    'scientific_name': 'Mangifera indica',
                    'optimal_temp_min': 10.0,
                    'optimal_temp_max': 13.0,
                    'optimal_humidity_min': 85.0,
                    'optimal_humidity_max': 90.0,
                    'optimal_light_max': 100,
                    'optimal_co2_max': 400,
                    'shelf_life_days': 10,
                    'ethylene_sensitive': True,
                    'chilling_sensitive': True,
                },
                {
                    'name': 'Orange',
                    'scientific_name': 'Citrus sinensis',
                    'optimal_temp_min': 4.0,
                    'optimal_temp_max': 10.0,
                    'optimal_humidity_min': 85.0,
                    'optimal_humidity_max': 90.0,
                    'optimal_light_max': 100,
                    'optimal_co2_max': 400,
                    'shelf_life_days': 21,
                    'ethylene_sensitive': False,
                    'chilling_sensitive': False,
                },
                {
                    'name': 'Tomato',
                    'scientific_name': 'Solanum lycopersicum',
                    'optimal_temp_min': 10.0,
                    'optimal_temp_max': 15.0,
                    'optimal_humidity_min': 85.0,
                    'optimal_humidity_max': 90.0,
                    'optimal_light_max': 100,
                    'optimal_co2_max': 400,
                    'shelf_life_days': 14,
                    'ethylene_sensitive': True,
                    'chilling_sensitive': True,
                },
            ]
            
            for fruit_data in default_fruits:
                if not FruitType.objects.filter(name=fruit_data['name']).exists():
                    FruitType.objects.create(**fruit_data)
            
            print("Bika: Default fruit types checked/created")
            
        except (OperationalError, ProgrammingError) as e:
            # Database tables don't exist yet (migrations not run)
            print(f"Bika: Database not ready: {e}")
        except Exception as e:
            print(f"Bika: Error initializing default data: {e}")