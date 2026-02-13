import os
from pathlib import Path
from datetime import timedelta

# -------------------------
# Base
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure-bika-project-secret-key-2025-change-this-in-production'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0','10.0.2.2', '172.16.16.195']

# -------------------------
# Installed Apps
# -------------------------
INSTALLED_APPS = [
    # Django
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',

    # Local apps
    'bika.apps.BikaConfig',
    'users.apps.UsersConfig',
    'products',

    # Third-party
    'rest_framework',
    'corsheaders',
    
]

# Optional apps (conditional import)
try:
    import crispy_forms
    INSTALLED_APPS.append('crispy_forms')
    try:
        import crispy_bootstrap4
        INSTALLED_APPS.append('crispy_bootstrap4')
        CRISPY_TEMPLATE_PACK = 'bootstrap4'
    except ImportError:
        try:
            import crispy_bootstrap3
            INSTALLED_APPS.append('crispy_bootstrap3')
            CRISPY_TEMPLATE_PACK = 'bootstrap3'
        except ImportError:
            CRISPY_TEMPLATE_PACK = 'uni_form'
except ImportError:
    pass

try:
    import django_extensions
    INSTALLED_APPS.append('django_extensions')
except ImportError:
    pass

try:
    import mathfilters
    INSTALLED_APPS.append('mathfilters')
except ImportError:
    pass

# -------------------------
# Middleware
# -------------------------
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # must be first for Flutter
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'bika.middleware.SecurityHeadersMiddleware',
    'bika.middleware.SessionTimeoutMiddleware',
    'bika.middleware.RoleBasedAccessMiddleware',
]

# -------------------------
# URLs and WSGI
# -------------------------
ROOT_URLCONF = 'bika_project.urls'
WSGI_APPLICATION = 'bika_project.wsgi.application'

# -------------------------
# Templates
# -------------------------
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
                'django.template.context_processors.static',
                'bika.context_processors.site_info',
                'bika.context_processors.cart_details',
                'bika.context_processors.user_profile_info',
            ],
        },
    },
]

# -------------------------
# Database
# -------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# -------------------------
# Password Validation
# -------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# -------------------------
# Internationalization
# -------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# -------------------------
# Static and Media
# -------------------------
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static', BASE_DIR / 'bika' / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# -------------------------
# Custom User
# -------------------------
AUTH_USER_MODEL = 'bika.CustomUser'
LOGIN_URL = 'bika:login'
LOGIN_REDIRECT_URL = 'bika:home'
LOGOUT_REDIRECT_URL = 'bika:home'

# -------------------------
# Django REST Framework
# -------------------------
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ),
}

# -------------------------
# Simple JWT
# -------------------------
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'AUTH_HEADER_TYPES': ('Bearer',),
}

# -------------------------
# Email
# -------------------------
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'abeliniyigena@gmail.com'
EMAIL_HOST_PASSWORD = 'MUGO12ruku__'
DEFAULT_FROM_EMAIL = 'Bika <noreply@bika.com>'

# -------------------------
# Sessions & CSRF
# -------------------------
SESSION_COOKIE_AGE = 3600
SESSION_SAVE_EVERY_REQUEST = True
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = False

CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://0.0.0.0:8000',
    'http://172.16.16.195:8000',
]

# -------------------------
# CORS for Flutter
# -------------------------
CORS_ALLOWED_ORIGINS = [
    'http://localhost:8000',
    'http://10.0.2.2:8000',
]

# -------------------------
# Cache
# -------------------------
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# -------------------------
# File Upload Limits
# -------------------------
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760

# -------------------------
# BIKA Settings
# -------------------------
BIKA_SETTINGS = {
    'APP_NAME': 'Bika',
    'APP_DESCRIPTION': 'AI-Powered Fruit Quality Monitoring & E-commerce Platform',
    'APP_VERSION': '1.0.0',
    'SUPPORT_EMAIL': 'support@bika.com',
    'SALES_EMAIL': 'sales@bika.com',
    'ADMIN_EMAIL': 'admin@bika.com',
    'MAX_PRODUCT_IMAGES': 10,
    'DEFAULT_STOCK_THRESHOLD': 5,
    'DEFAULT_TAX_RATE': 0.18,
    'SHIPPING_COST': 5000,
    'FREE_SHIPPING_THRESHOLD': 100000,
    'ORDER_PROCESSING_DAYS': 1,
    'DELIVERY_ESTIMATE_DAYS': 3,
    'DEFAULT_FRUIT_SHELF_LIFE': 7,
    'QUALITY_CHECK_INTERVAL_HOURS': 24,
    'CRITICAL_TEMP_THRESHOLD': 10,
    'CRITICAL_HUMIDITY_THRESHOLD': 95,
}

BIKA_AI_SERVICE_TYPE = 'enhanced'
BIKA_AI_MODEL_DIR = os.path.join(MEDIA_ROOT, 'fruit_models')
BIKA_AI_CACHE_TIMEOUT = 3600
BIKA_AI_MAX_PREDICTIONS_PER_BATCH = 1000

# -------------------------
# Create Required Directories
# -------------------------
required_dirs = [
    MEDIA_ROOT,
    MEDIA_ROOT / 'products',
    MEDIA_ROOT / 'categories',
    MEDIA_ROOT / 'profiles',
    MEDIA_ROOT / 'business_logos',
    MEDIA_ROOT / 'trained_models',
    MEDIA_ROOT / 'datasets',
    MEDIA_ROOT / 'fruit_datasets',
    MEDIA_ROOT / 'fruit_models',
    MEDIA_ROOT / 'services',
    MEDIA_ROOT / 'testimonials',
    MEDIA_ROOT / 'fruits',
    MEDIA_ROOT / 'site' / 'logo',
    MEDIA_ROOT / 'site' / 'favicon',
    STATIC_ROOT,
    BASE_DIR / 'logs',
]

for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# -------------------------
# Security
# -------------------------
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# -------------------------
# Error Handlers
# -------------------------
handler404 = 'bika.views.handler404'
handler500 = 'bika.views.handler500'
handler403 = 'bika.views.handler403'
handler400 = 'bika.views.handler400'
