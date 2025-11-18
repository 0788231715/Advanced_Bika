from .models import SiteInfo

def site_info(request):
    """Add site information to all templates"""
    try:
        site_info = SiteInfo.objects.first()
        if not site_info:
            # Create default site info
            site_info = SiteInfo.objects.create(
                name="Bika",
                tagline="Your Success Is Our Business",
                description="Bika provides exceptional services to help your business grow.",
                email="contact@bika.com"
            )
    except Exception:
        site_info = None
    
    return {
        'site_info': site_info
    }