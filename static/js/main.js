// Bika Main JavaScript File
// Contains all global functionality for the Bika platform

class BikaApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupGlobalHandlers();
        this.setupAJAX();
        this.setupAnimations();
        this.setupUtilities();
        console.log('🎯 Bika App Initialized');
    }

    // Event Listeners Setup
    setupEventListeners() {
        // Global click handlers
        this.setupGlobalClicks();
        
        // Form handlers
        this.setupFormHandlers();
        
        // Navigation handlers
        this.setupNavigationHandlers();
        
        // UI interaction handlers
        this.setupUIHandlers();
    }

    // Global Click Handlers
    setupGlobalClicks() {
        // Smooth scrolling for all anchor links
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href^="#"]');
            if (link) {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.dropdown')) {
                document.querySelectorAll('.dropdown-menu.show').forEach(menu => {
                    const dropdown = bootstrap.Dropdown.getInstance(menu.previousElementSibling);
                    if (dropdown) dropdown.hide();
                });
            }
        });
    }

    // Form Handlers
    setupFormHandlers() {
        // Auto-dismiss alerts
        this.setupAutoDismissAlerts();

        // Form validation enhancement
        this.setupFormValidation();

        // Contact form handling
        this.setupContactForm();

        // Newsletter form handling
        this.setupNewsletterForm();
    }

    // Auto-dismiss alerts after 5 seconds
    setupAutoDismissAlerts() {
        const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(alert => {
            setTimeout(() => {
                if (alert.parentNode) {
                    bootstrap.Alert.getOrCreateInstance(alert).close();
                }
            }, 5000);
        });
    }

    // Enhanced form validation
    setupFormValidation() {
        const forms = document.querySelectorAll('.needs-validation');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                form.classList.add('was-validated');
            }, false);
        });
    }

    // Contact form handling
    setupContactForm() {
        const contactForm = document.getElementById('contactForm');
        if (contactForm) {
            contactForm.addEventListener('submit', (e) => {
                this.handleFormSubmission(e, contactForm, 'contact');
            });
        }
    }

    // Newsletter form handling
    setupNewsletterForm() {
        const newsletterForms = document.querySelectorAll('.newsletter-form');
        newsletterForms.forEach(form => {
            form.addEventListener('submit', (e) => {
                this.handleFormSubmission(e, form, 'newsletter');
            });
        });
    }

    // Generic form submission handler
    async handleFormSubmission(e, form, formType) {
        e.preventDefault();
        
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        
        // Show loading state
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        submitBtn.disabled = true;

        try {
            const formData = new FormData(form);
            
            // Add CSRF token if not present
            if (!formData.has('csrfmiddlewaretoken')) {
                const csrfToken = this.getCSRFToken();
                if (csrfToken) {
                    formData.append('csrfmiddlewaretoken', csrfToken);
                }
            }

            const response = await fetch(form.action || window.location.href, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.showNotification(data.message || 'Success!', 'success');
                form.reset();
                form.classList.remove('was-validated');
            } else {
                this.showNotification(data.message || 'An error occurred.', 'error');
            }

        } catch (error) {
            console.error('Form submission error:', error);
            this.showNotification('Network error. Please try again.', 'error');
        } finally {
            // Restore button state
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    }

    // Navigation Handlers
    setupNavigationHandlers() {
        // Active navigation highlighting
        this.highlightActiveNav();

        // Mobile menu enhancements
        this.setupMobileMenu();

        // Sticky header behavior
        this.setupStickyHeader();
    }

    // Highlight active navigation
    highlightActiveNav() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            const linkPath = link.getAttribute('href');
            if (linkPath === currentPath || 
                (linkPath !== '/' && currentPath.startsWith(linkPath)) ||
                (linkPath && linkPath.includes('services') && currentPath.includes('services'))) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    // Mobile menu enhancements
    setupMobileMenu() {
        const navbarToggler = document.querySelector('.navbar-toggler');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        
        if (navbarToggler && navbarCollapse) {
            // Close mobile menu when clicking on a link
            const navLinks = navbarCollapse.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                link.addEventListener('click', () => {
                    if (window.innerWidth < 992) {
                        const bsCollapse = bootstrap.Collapse.getInstance(navbarCollapse);
                        if (bsCollapse) bsCollapse.hide();
                    }
                });
            });
        }
    }

    // Sticky header behavior
    setupStickyHeader() {
        const header = document.querySelector('.navbar');
        if (header) {
            window.addEventListener('scroll', () => {
                if (window.scrollY > 100) {
                    header.classList.add('scrolled');
                } else {
                    header.classList.remove('scrolled');
                }
            });
        }
    }

    // UI Interaction Handlers
    setupUIHandlers() {
        // Search functionality
        this.setupSearch();

        // Back to top button
        this.setupBackToTop();

        // Lazy loading images
        this.setupLazyLoading();

        // Counter animations
        this.setupCounters();
    }

    // Search functionality
    setupSearch() {
        const searchTriggers = document.querySelectorAll('.search-trigger');
        const searchOverlay = document.getElementById('searchOverlay');
        
        if (searchTriggers.length && searchOverlay) {
            searchTriggers.forEach(trigger => {
                trigger.addEventListener('click', () => {
                    searchOverlay.classList.add('show');
                    document.body.style.overflow = 'hidden';
                    
                    const searchInput = searchOverlay.querySelector('input[type="search"], input[type="text"]');
                    if (searchInput) searchInput.focus();
                });
            });

            // Close search overlay
            const closeBtn = searchOverlay.querySelector('.search-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    searchOverlay.classList.remove('show');
                    document.body.style.overflow = '';
                });
            }

            // Close on escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && searchOverlay.classList.contains('show')) {
                    searchOverlay.classList.remove('show');
                    document.body.style.overflow = '';
                }
            });

            // Close on overlay click
            searchOverlay.addEventListener('click', (e) => {
                if (e.target === searchOverlay) {
                    searchOverlay.classList.remove('show');
                    document.body.style.overflow = '';
                }
            });
        }
    }

    // Back to top button
    setupBackToTop() {
        const backToTop = document.createElement('button');
        backToTop.innerHTML = '<i class="fas fa-chevron-up"></i>';
        backToTop.className = 'btn btn-primary back-to-top';
        backToTop.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: none;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
            transition: all 0.3s ease;
        `;

        backToTop.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });

        backToTop.addEventListener('mouseenter', () => {
            backToTop.style.transform = 'translateY(-3px)';
        });

        backToTop.addEventListener('mouseleave', () => {
            backToTop.style.transform = 'translateY(0)';
        });

        document.body.appendChild(backToTop);

        // Show/hide based on scroll position
        window.addEventListener('scroll', () => {
            if (window.scrollY > 300) {
                backToTop.style.display = 'flex';
            } else {
                backToTop.style.display = 'none';
            }
        });
    }

    // Lazy loading for images
    setupLazyLoading() {
        if ('IntersectionObserver' in window) {
            const lazyImages = document.querySelectorAll('img[data-src]');
            
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        imageObserver.unobserve(img);
                    }
                });
            });

            lazyImages.forEach(img => imageObserver.observe(img));
        }
    }

    // Counter animations
    setupCounters() {
        const counters = document.querySelectorAll('.counter');
        if (counters.length) {
            const counterObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.animateCounter(entry.target);
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.5 });

            counters.forEach(counter => counterObserver.observe(counter));
        }
    }

    // Animate counter numbers
    animateCounter(element) {
        const target = parseInt(element.getAttribute('data-target') || element.textContent);
        const duration = 2000; // 2 seconds
        const step = target / (duration / 16); // 60fps
        let current = 0;

        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                element.textContent = this.formatNumber(target);
                clearInterval(timer);
            } else {
                element.textContent = this.formatNumber(Math.floor(current));
            }
        }, 16);
    }

    // Format numbers with commas
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    // Global Handlers
    setupGlobalHandlers() {
        // Error handling
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
        });

        // Resize handlers
        this.setupResizeHandlers();

        // Online/offline detection
        this.setupConnectivityDetection();
    }

    // Resize handlers
    setupResizeHandlers() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 250);
        });
    }

    handleResize() {
        // Update any layout-dependent functionality
        this.highlightActiveNav();
    }

    // Connectivity detection
    setupConnectivityDetection() {
        window.addEventListener('online', () => {
            this.showNotification('Connection restored', 'success', 3000);
        });

        window.addEventListener('offline', () => {
            this.showNotification('You are currently offline', 'warning', 0);
        });
    }

    // AJAX Setup
    setupAJAX() {
        // Global AJAX settings
        this.setupAJAXDefaults();
    }

    setupAJAXDefaults() {
        // Set up CSRF token for all AJAX requests
        const csrfToken = this.getCSRFToken();
        if (csrfToken) {
            $.ajaxSetup({
                headers: {
                    'X-CSRFToken': csrfToken
                }
            });
        }
    }

    // Get CSRF token from cookies
    getCSRFToken() {
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Animations Setup
    setupAnimations() {
        // Scroll animations
        this.setupScrollAnimations();

        // Hover animations
        this.setupHoverAnimations();
    }

    // Scroll animations
    setupScrollAnimations() {
        if ('IntersectionObserver' in window) {
            const animatedElements = document.querySelectorAll('.animate-on-scroll');
            
            const animationObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animated');
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.1 });

            animatedElements.forEach(element => animationObserver.observe(element));
        }
    }

    // Hover animations
    setupHoverAnimations() {
        // Add hover effects to cards
        const cards = document.querySelectorAll('.card, .service-card, .feature-card');
        cards.forEach(card => {
            card.style.transition = 'all 0.3s ease';
            
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-5px)';
                card.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.15)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = '';
            });
        });
    }

    // Utilities Setup
    setupUtilities() {
        // Utility functions
        this.setupUtilityFunctions();
    }

    setupUtilityFunctions() {
        // Debounce function
        window.debounce = (func, wait) => {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        };

        // Throttle function
        window.throttle = (func, limit) => {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        };
    }

    // Notification system
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            border: none;
            border-radius: 10px;
        `;
        
        const icon = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        }[type] || 'info-circle';

        notification.innerHTML = `
            <i class="fas fa-${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentNode) {
                    bootstrap.Alert.getOrCreateInstance(notification).close();
                }
            }, duration);
        }

        return notification;
    }

    // Public methods
    refresh() {
        this.highlightActiveNav();
    }

    showLoading() {
        this.showNotification('Loading...', 'info', 0);
    }

    hideLoading() {
        document.querySelectorAll('.alert').forEach(alert => {
            if (alert.textContent.includes('Loading...')) {
                bootstrap.Alert.getOrCreateInstance(alert).close();
            }
        });
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.bikaApp = new BikaApp();
});

// Export for global access
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BikaApp;
}