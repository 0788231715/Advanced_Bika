# bika/urls.py - UPDATED AND OPTIMIZED VERSION
from django.urls import include, path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'bika'

urlpatterns = [
    # ==================== MAIN PAGES ====================
    path('', views.HomeView.as_view(), name='home'),
    path('about/', views.about_view, name='about'),
    path('services/', views.services_view, name='services'),
    path('services/<slug:slug>/', views.ServiceDetailView.as_view(), name='service_detail'),
    path('contact/', views.contact_view, name='contact'),
    path('faq/', views.faq_view, name='faq'),
    
    # ==================== AUTHENTICATION ====================
    path('login/', auth_views.LoginView.as_view(
        template_name='bika/pages/registration/login.html',
        redirect_authenticated_user=True
    ), name='login'),
    
    path('logout/', views.custom_logout, name='logout'),
    path('logout/success/', views.logout_success, name='logout_success'),
    
    path('register/', views.register_view, name='register'),
    path('vendor/register/', views.vendor_register_view, name='vendor_register'),
    
    # ==================== PASSWORD RESET ====================
    path('password-reset/', 
         auth_views.PasswordResetView.as_view(
             template_name='bika/pages/registration/password_reset.html',
             email_template_name='bika/pages/registration/password_reset_email.html',
             subject_template_name='bika/pages/registration/password_reset_subject.txt',
             success_url='/password-reset/done/'
         ), 
         name='password_reset'),
    
    path('password-reset/done/', 
         auth_views.PasswordResetDoneView.as_view(
             template_name='bika/pages/registration/password_reset_done.html'
         ), 
         name='password_reset_done'),
    
    path('password-reset-confirm/<uidb64>/<token>/', 
         auth_views.PasswordResetConfirmView.as_view(
             template_name='bika/pages/registration/password_reset_confirm.html',
             success_url='/password-reset-complete/'
         ), 
         name='password_reset_confirm'),
    
    path('password-reset-complete/', 
         auth_views.PasswordResetCompleteView.as_view(
             template_name='bika/pages/registration/password_reset_complete.html'
         ), 
         name='password_reset_complete'),
    
    # ==================== ADMIN DASHBOARD ====================
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/register-client/', views.register_client, name='register_client'),
    # In your urlpatterns
    path('api/dashboard/stats/', views.dashboard_stats_api, name='dashboard_stats_api'),
    # ==================== PRODUCTS ====================
    path('products/', views.product_list_view, name='product_list'),
    path('products/category/<slug:category_slug>/', views.products_by_category_view, name='products_by_category'),
    path('products/<slug:slug>/', views.product_detail_view, name='product_detail'),
    path('products/<int:product_id>/buy-now/', views.buy_now_product, name='buy_now_product'),
    path('products/compare/', views.product_comparison_view, name='product_comparison'),
    path('products/search/', views.product_search_view, name='product_search'),
    path('products/<int:product_id>/review/', views.add_review, name='add_review'),
    
    # ==================== VENDOR ====================
    path('vendor/dashboard/', views.vendor_dashboard, name='vendor_dashboard'),
    path('vendor/products/', views.vendor_product_list, name='vendor_product_list'),
    path('vendor/products/add/', views.vendor_add_product, name='vendor_add_product'),
    path('vendor/products/add-for-client/', views.add_client_product, name='add_client_product'),
    path('vendor/products/edit/<int:product_id>/', views.vendor_edit_product, name='vendor_edit_product'),
    path('vendor/products/delete/<int:product_id>/', views.vendor_delete_product, name='vendor_delete_product'),
    path('vendor/track-products/', views.track_my_products, name='track_my_products'),
    
    # ==================== USER PROFILE ====================
    path('profile/', views.user_profile, name='user_profile'),
    path('profile/update/', views.update_profile, name='update_profile'),
    path('profile/settings/', views.user_settings, name='user_settings'),
    
    # ==================== ORDERS ====================
    path('orders/', views.user_orders, name='user_orders'),
    
    # Order detail and actions - using include for cleaner structure
    path('orders/<int:order_id>/', include([
        path('', views.order_detail, name='order_detail'),
        path('cancel/', views.cancel_order, name='cancel_order'),
        path('review/', views.create_review, name='create_review'),
    ])),
    
    # ==================== USER ADDRESSES ====================
    path('user/addresses/', views.user_address_book_view, name='user_address_book'),
    path('user/addresses/add/', views.user_add_edit_address_view, name='user_add_address'),
    path('user/addresses/edit/<int:address_id>/', views.user_add_edit_address_view, name='user_edit_address'),
    path('user/addresses/delete/<int:address_id>/', views.user_delete_address_view, name='user_delete_address'),
    
    # ==================== CART ====================
    path('cart/', views.cart, name='cart'),
    path('cart/add/<int:product_id>/', views.add_to_cart, name='add_to_cart'),
    path('cart/quick-add/<int:product_id>/', views.quick_add_to_cart, name='quick_add_to_cart'),
    path('cart/update/<int:product_id>/', views.update_cart, name='update_cart'),
    path('cart/remove/<int:product_id>/', views.remove_from_cart, name='remove_from_cart'),
    path('cart/clear/', views.clear_cart, name='clear_cart'),
    
    # ==================== WISHLIST ====================
    path('wishlist/', views.wishlist, name='wishlist'),
    path('wishlist/add/<int:product_id>/', views.add_to_wishlist, name='add_to_wishlist'),
    path('wishlist/remove/<int:product_id>/', views.remove_from_wishlist, name='remove_from_wishlist'),
    
    # ==================== CHECKOUT & PAYMENT ====================
    path('checkout/', views.checkout, name='checkout'),
    path('checkout/place-order/', views.place_order, name='place_order'),
    path('checkout/order-confirmation/<int:order_id>/', views.order_confirmation, name='order_confirmation'),
    path('payment/<int:payment_id>/', views.payment_processing, name='payment_processing'),
    path('api/payment/webhook/', views.payment_webhook, name='payment_webhook'),
    
    # ==================== FRUIT QUALITY MONITORING ====================
    path('fruit-quality/dashboard/', views.fruit_quality_dashboard, name='fruit_quality_dashboard'),
    path('fruit-quality/batches/create/', views.create_fruit_batch, name='create_fruit_batch'),
    path('fruit-quality/batches/<int:batch_id>/', views.batch_detail, name='batch_detail'),
    path('fruit-quality/batches/<int:batch_id>/add-reading/', views.add_quality_reading, name='add_quality_reading'),
    path('fruit-quality/batches/<int:batch_id>/analytics/', views.batch_analytics, name='batch_analytics'),
    
    # ==================== NOTIFICATIONS ====================
    path('notifications/', views.notifications, name='notifications'),
    path('notifications/<int:notification_id>/read/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),
    path('api/notifications/unread-count/', views.unread_notifications_count, name='unread_notifications_count'),
    
    # ==================== STORAGE & TRACKING ====================
    path('admin/storage-sites/', views.storage_sites, name='storage_sites'),
    path('scan/', views.scan_product, name='scan_product'),
    
    # ==================== API ENDPOINTS ====================
    # Product API
    path('api/product/<str:barcode>/', views.api_product_detail, name='api_product_detail'),
    path('api/delivery-status/<int:delivery_id>/', views.api_delivery_status, name='api_delivery_status'),
    path('api/delivery/update-status/', views.api_update_delivery_status, name='api_update_delivery_status'), # NEW
    path('api/products/<int:product_id>/analytics/', views.product_analytics_api, name='product_analytics_api'),
    path('api/delivery/update-status/', views.update_delivery_status, name='api_update_delivery_status'),
    path('api/upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('api/inventory/move/', views.storage_check_in, name='api_record_inventory_move'),
    path('api/train-model/', views.train_model, name='train_model'),
    path('api/sensor-data/', views.receive_sensor_data, name='receive_sensor_data'),
    path('api/train-fruit-model/', views.train_fruit_model_api, name='train_fruit_model'),
    path('api/predict-fruit-quality/', views.predict_fruit_quality_api, name='predict_fruit_quality'),
    path('api/warehouse/layout/<int:location_id>/', views.api_get_warehouse_layout, name='api_get_warehouse_layout'), # NEW
    path('api/storage-compatibility/', views.storage_compatibility_check, name='storage_compatibility'),
    
    # Alerts API
    path('api/alerts/<int:alert_id>/resolve/', views.resolve_alert, name='resolve_alert'),
    
    # Newsletter
    path('api/newsletter/subscribe/', views.newsletter_subscribe, name='newsletter_subscribe'),
    
    # ==================== UTILITY ====================
    path('vendor/products/bulk-action/', views.handle_bulk_actions, name='handle_bulk_actions'),
    
    # Dashboard API endpoints
    path('api/dashboard/sales-analytics/', views.sales_analytics_api, name='sales_analytics_api'),
    path('api/dashboard/messages-analytics/', views.messages_analytics_api, name='messages_analytics_api'),
    path('api/dashboard/alerts/', views.get_active_alerts, name='get_active_alerts'),
    path('api/dashboard/performance/', views.performance_metrics_api, name='performance_metrics_api'),
    path('api/dashboard/export-inventory/', views.export_inventory_report, name='export_inventory_report'),
    path('api/dashboard/user-activity/', views.get_user_activity, name='get_user_activity'),
    
    # ==================== AI ALERT SYSTEM ====================
    path('admin/ai-alerts/', views.ai_alert_dashboard, name='ai_alert_dashboard'),
    path('admin/scan-products/', views.scan_all_products_for_alerts, name='scan_products'),
    path('admin/train-new-model/', views.train_new_model_view, name='train_new_model'),
    path('admin/model-management/', views.model_management, name='model_management'),
    path('admin/activate-model/<int:model_id>/', views.activate_model, name='activate_model'),
    path('admin/generate-sample-data/', views.generate_sample_data_view, name='generate_sample_data'),
    path('admin/download-dataset/', views.download_generated_dataset, name='download_dataset'),
    
    # ==================== PRODUCT AI INSIGHTS ====================
    path('admin/product-ai-insights-overview/', views.product_ai_insights_overview, name='product_ai_insights_overview'),
    path('product/<int:product_id>/ai-insights/', views.product_ai_insights, name='product_ai_insights'),
    
    # ==================== ADDITIONAL API ENDPOINTS ====================
    path('api/batch-scan/', views.batch_product_scan_api, name='batch_scan_api'),
    path('api/product/<int:product_id>/quality-prediction/', views.get_product_quality_prediction, name='quality_prediction_api'),
    path('api/analyze-csv/', views.analyze_csv, name='analyze_csv'),
    path('api/inventory/move/', views.api_record_inventory_movement, name='api_record_inventory_move'), # NEW
    path('api/warehouse/layout/<int:location_id>/', views.api_get_warehouse_layout, name='api_get_warehouse_layout'), # NEW
    
    # ==================== AI TRAINING ====================
    path('ai/train-models/', views.train_five_models_view, name='train_models'),
    path('ai/training-results/', views.training_results_view, name='training_results'),
    path('ai/model-comparison/', views.model_comparison_view, name='model_comparison'),
    path('ai/generate-sample-dataset/', views.generate_sample_dataset_view, name='generate_sample_dataset'),
    
    # ==================== FAVICON (to avoid 404) ====================
    path('favicon.ico', views.favicon_view, name='favicon'),
    
    # ==================== DEBUG & UTILITY ====================
    path('debug/urls/', views.debug_urls, name='debug_urls'),
    path('api/dashboard/export-sales/', views.export_sales_report, name='export_sales_report'),

    # ==================== ROLE-BASED URLS ====================
    # Manager URLs
    path('manager/', views.manager_dashboard, name='manager_dashboard'),
    path('manager/inventory/', views.manager_inventory, name='manager_inventory'),
    path('manager/deliveries/', views.manager_deliveries, name='manager_deliveries'),
    path('manager/deliveries/<int:delivery_id>/', views.manager_delivery_detail_view, name='manager_delivery_detail'),
    path('manager/reports/', views.manager_reports, name='manager_reports'),
    path('manager/team/', views.manager_team_view, name='manager_team'),
    path('manager/deliveries/<int:delivery_id>/update-status/', views.update_delivery_status, name='update_delivery_status'),
    path('manager/deliveries/<int:delivery_id>/assign-staff/', views.assign_delivery_staff, name='assign_delivery_staff'),
    path('manager/roles/', views.manager_user_role_management, name='manager_user_role_management'),
    path('manager/roles/edit/<int:user_id>/', views.manager_user_role_management, name='manager_user_role_management_edit'),

    # Storage Staff URLs
    path('storage/', views.storage_dashboard, name='storage_dashboard'),
    path('storage/inventory/', views.storage_inventory, name='storage_inventory'),
    path('storage/locations/', views.storage_locations, name='storage_locations'),
    path('storage/check-in/', views.storage_check_in, name='storage_check_in'),
    path('storage/check-out/', views.storage_check_out, name='storage_check_out'),
    path('storage/inventory/add/', views.storage_add_edit_inventory_item_view, name='storage_add_inventory_item'),
    path('storage/inventory/edit/<int:item_id>/', views.storage_add_edit_inventory_item_view, name='storage_edit_inventory_item'),
    path('storage/transfer/', views.storage_transfer, name='storage_transfer'),

    # Client URLs
    path('client/dashboard/', views.client_dashboard, name='client_dashboard'),
    path('client/inventory/', views.client_inventory, name='client_inventory'),
    path('client/inventory/<int:item_id>/', views.client_item_detail, name='client_item_detail'),
    path('client/deliveries/', views.client_deliveries, name='client_deliveries'),
    path('client/deliveries/<int:delivery_id>/', views.client_delivery_detail, name='client_delivery_detail'),
    path('client/deliveries/<int:delivery_id>/track-map/', views.track_order_map, name='track_order_map'),
    path('client/requests/', views.client_requests, name='client_requests'),
    path('client/requests/create/', views.create_client_request, name='create_client_request'),
    path('client/requests/', views.client_requests_list, name='client_requests_list'),
    path('client/requests/<int:request_id>/', views.client_request_detail, name='client_request_detail'),
    
]

# Add URL patterns for Fruit Quality Dashboard (admin)
urlpatterns += [
    path('admin/fruit-dashboard/', views.fruit_quality_dashboard, name='fruit_dashboard'),
]