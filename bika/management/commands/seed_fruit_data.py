from django.core.management.base import BaseCommand
from bika.models import FruitType
from datetime import datetime

class Command(BaseCommand):
    help = 'Seed initial fruit data'
    
    def handle(self, *args, **options):
        fruits_data = [
            {
                'name': 'Banana',
                'scientific_name': 'Musa acuminata',
                'optimal_temp_min': 13,
                'optimal_temp_max': 15,
                'optimal_humidity_min': 85,
                'optimal_humidity_max': 95,
                'optimal_light_max': 100,
                'optimal_co2_max': 400,
                'shelf_life_days': 7,
                'ethylene_sensitive': False,
                'chilling_sensitive': True
            },
            {
                'name': 'Apple',
                'scientific_name': 'Malus domestica',
                'optimal_temp_min': 0,
                'optimal_temp_max': 4,
                'optimal_humidity_min': 90,
                'optimal_humidity_max': 95,
                'optimal_light_max': 50,
                'optimal_co2_max': 350,
                'shelf_life_days': 30,
                'ethylene_sensitive': True,
                'chilling_sensitive': False
            },
            {
                'name': 'Orange',
                'scientific_name': 'Citrus × sinensis',
                'optimal_temp_min': 5,
                'optimal_temp_max': 10,
                'optimal_humidity_min': 85,
                'optimal_humidity_max': 90,
                'optimal_light_max': 100,
                'optimal_co2_max': 400,
                'shelf_life_days': 21,
                'ethylene_sensitive': False,
                'chilling_sensitive': True
            },
            {
                'name': 'Tomato',
                'scientific_name': 'Solanum lycopersicum',
                'optimal_temp_min': 10,
                'optimal_temp_max': 15,
                'optimal_humidity_min': 85,
                'optimal_humidity_max': 90,
                'optimal_light_max': 100,
                'optimal_co2_max': 400,
                'shelf_life_days': 14,
                'ethylene_sensitive': True,
                'chilling_sensitive': True
            },
            {
                'name': 'Mango',
                'scientific_name': 'Mangifera indica',
                'optimal_temp_min': 10,
                'optimal_temp_max': 15,
                'optimal_humidity_min': 85,
                'optimal_humidity_max': 90,
                'optimal_light_max': 100,
                'optimal_co2_max': 400,
                'shelf_life_days': 10,
                'ethylene_sensitive': True,
                'chilling_sensitive': True
            },
            {
                'name': 'Strawberry',
                'scientific_name': 'Fragaria × ananassa',
                'optimal_temp_min': 0,
                'optimal_temp_max': 2,
                'optimal_humidity_min': 90,
                'optimal_humidity_max': 95,
                'optimal_light_max': 50,
                'optimal_co2_max': 350,
                'shelf_life_days': 5,
                'ethylene_sensitive': False,
                'chilling_sensitive': False
            },
        ]
        
        for fruit_data in fruits_data:
            fruit, created = FruitType.objects.get_or_create(
                name=fruit_data['name'],
                defaults=fruit_data
            )
            
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Created {fruit.name}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'{fruit.name} already exists')
                )