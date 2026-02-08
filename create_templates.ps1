Write-Host "Creating Bika Django Template Structure..." -ForegroundColor Green

# Create all directories
$directories = @(
    # Templates directories
    "bika\templates\bika",
    "bika\templates\bika\partials",
    "bika\templates\bika\pages",
    "bika\templates\bika\pages\error_pages",
    "bika\templates\bika\pages\admin",
    "bika\templates\bika\pages\vendor",
    "bika\templates\bika\pages\user",
    "bika\templates\bika\pages\manager",
    "bika\templates\bika\pages\storage",
    "bika\templates\bika\pages\client",
    "bika\templates\bika\pages\registration",
    "bika\templates\bika\pages\ai",
    "bika\templates\bika\email",
    "bika\templates\bika\includes",
    "bika\templates\bika\includes\modals",
    "bika\templates\bika\includes\widgets",
    
    # Static directories
    "bika\static\bika",
    "bika\static\bika\css",
    "bika\static\bika\css\theme",
    "bika\static\bika\js",
    "bika\static\bika\js\lib",
    "bika\static\bika\js\lib\chartjs",
    "bika\static\bika\js\lib\datatables",
    "bika\static\bika\images",
    "bika\static\bika\images\hero",
    "bika\static\bika\images\products",
    "bika\static\bika\images\categories",
    "bika\static\bika\images\users",
    "bika\static\bika\images\icons",
    "bika\static\bika\images\icons\fruits",
    "bika\static\bika\images\icons\payment",
    "bika\static\bika\images\icons\social",
    "bika\static\bika\fonts",
    "bika\static\bika\media",
    "bika\static\bika\media\product_images",
    "bika\static\bika\media\user_uploads",
    "bika\static\bika\media\datasets",
    "bika\static\bika\media\trained_models"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "Created directory: $dir" -ForegroundColor Cyan
}

Write-Host "`nCreating template files..." -ForegroundColor Yellow

# List of all template files to create
$templateFiles = @(
    # Main templates
    "bika\templates\bika\base.html",
    "bika\templates\bika\index.html",
    
    # Partials
    "bika\templates\bika\partials\navbar.html",
    "bika\templates\bika\partials\footer.html",
    "bika\templates\bika\partials\messages.html",
    "bika\templates\bika\partials\sidebar.html",
    "bika\templates\bika\partials\pagination.html",
    "bika\templates\bika\partials\product_card.html",
    "bika\templates\bika\partials\breadcrumb.html",
    "bika\templates\bika\partials\notifications_dropdown.html",
    
    # Pages
    "bika\templates\bika\pages\home.html",
    "bika\templates\bika\pages\about.html",
    "bika\templates\bika\pages\contact.html",
    "bika\templates\bika\pages\faq.html",
    "bika\templates\bika\pages\services.html",
    "bika\templates\bika\pages\service_detail.html",
    "bika\templates\bika\pages\products.html",
    "bika\templates\bika\pages\product_detail.html",
    "bika\templates\bika\pages\products_by_category.html",
    "bika\templates\bika\pages\search_results.html",
    "bika\templates\bika\pages\scan_product.html",
    "bika\templates\bika\pages\maintenance.html",
    
    # Error pages
    "bika\templates\bika\pages\error_pages\404.html",
    "bika\templates\bika\pages\error_pages\500.html",
    "bika\templates\bika\pages\error_pages\403.html",
    "bika\templates\bika\pages\error_pages\400.html",
    
    # Admin pages
    "bika\templates\bika\pages\admin\dashboard.html",
    "bika\templates\bika\pages\admin\ai_alert_dashboard.html",
    "bika\templates\bika\pages\admin\train_new_model.html",
    "bika\templates\bika\pages\admin\model_management.html",
    "bika\templates\bika\pages\admin\training_results.html",
    "bika\templates\bika\pages\admin\generate_sample_data.html",
    "bika\templates\bika\pages\admin\storage_sites.html",
    "bika\templates\bika\pages\admin\fruit_dashboard.html",
    "bika\templates\bika\pages\admin\product_ai_insights_overview.html",
    
    # Vendor pages
    "bika\templates\bika\pages\vendor\dashboard.html",
    "bika\templates\bika\pages\vendor\products.html",
    "bika\templates\bika\pages\vendor\add_product.html",
    "bika\templates\bika\pages\vendor\edit_product.html",
    "bika\templates\bika\pages\vendor\fruit_dashboard.html",
    "bika\templates\bika\pages\vendor\batch_detail.html",
    "bika\templates\bika\pages\vendor\create_fruit_batch.html",
    "bika\templates\bika\pages\vendor\add_quality_reading.html",
    "bika\templates\bika\pages\vendor\batch_analytics.html",
    "bika\templates\bika\pages\vendor\track_products.html",
    "bika\templates\bika\pages\vendor\vendor_orders.html",
    
    # User pages
    "bika\templates\bika\pages\user\profile.html",
    "bika\templates\bika\pages\user\settings.html",
    "bika\templates\bika\pages\user\orders.html",
    "bika\templates\bika\pages\user\order_detail.html",
    "bika\templates\bika\pages\user\create_review.html",
    "bika\templates\bika\pages\user\wishlist.html",
    "bika\templates\bika\pages\user\cart.html",
    "bika\templates\bika\pages\user\checkout.html",
    "bika\templates\bika\pages\user\payment_processing.html",
    "bika\templates\bika\pages\user\notifications.html",
    "bika\templates\bika\pages\user\change_password.html",
    
    # Manager pages
    "bika\templates\bika\pages\manager\dashboard.html",
    "bika\templates\bika\pages\manager\inventory.html",
    "bika\templates\bika\pages\manager\deliveries.html",
    "bika\templates\bika\pages\manager\reports.html",
    "bika\templates\bika\pages\manager\team.html",
    
    # Storage pages
    "bika\templates\bika\pages\storage\dashboard.html",
    "bika\templates\bika\pages\storage\inventory.html",
    "bika\templates\bika\pages\storage\locations.html",
    "bika\templates\bika\pages\storage\check_in.html",
    "bika\templates\bika\pages\storage\check_out.html",
    "bika\templates\bika\pages\storage\transfer.html",
    
    # Client pages
    "bika\templates\bika\pages\client\dashboard.html",
    "bika\templates\bika\pages\client\inventory.html",
    "bika\templates\bika\pages\client\item_detail.html",
    "bika\templates\bika\pages\client\deliveries.html",
    "bika\templates\bika\pages\client\delivery_detail.html",
    "bika\templates\bika\pages\client\requests.html",
    "bika\templates\bika\pages\client\request_detail.html",
    "bika\templates\bika\pages\client\create_request.html",
    
    # Registration pages
    "bika\templates\bika\pages\registration\login.html",
    "bika\templates\bika\pages\registration\register.html",
    "bika\templates\bika\pages\registration\vendor_register.html",
    "bika\templates\bika\pages\registration\logout.html",
    "bika\templates\bika\pages\registration\password_reset.html",
    "bika\templates\bika\pages\registration\password_reset_confirm.html",
    
    # AI pages
    "bika\templates\bika\pages\ai\upload_dataset.html",
    "bika\templates\bika\pages\ai\train_model.html",
    "bika\templates\bika\pages\ai\model_comparison.html",
    "bika\templates\bika\pages\ai\predictions.html",
    "bika\templates\bika\pages\ai\analytics.html",
    
    # Email templates
    "bika\templates\bika\email\order_confirmation.html",
    "bika\templates\bika\email\welcome.html",
    "bika\templates\bika\email\password_reset.html",
    "bika\templates\bika\email\invoice.html",
    "bika\templates\bika\email\alert_notification.html",
    
    # Includes
    "bika\templates\bika\includes\head.html",
    "bika\templates\bika\includes\scripts.html",
    "bika\templates\bika\includes\modals\quick_view.html",
    "bika\templates\bika\includes\modals\add_to_cart.html",
    "bika\templates\bika\includes\modals\delete_confirmation.html",
    "bika\templates\bika\includes\widgets\cart_widget.html",
    "bika\templates\bika\includes\widgets\search_widget.html",
    "bika\templates\bika\includes\widgets\filter_widget.html",
    
    # CSS files
    "bika\static\bika\css\style.css",
    "bika\static\bika\css\admin.css",
    "bika\static\bika\css\vendor.css",
    "bika\static\bika\css\responsive.css",
    "bika\static\bika\css\theme\colors.css",
    "bika\static\bika\css\theme\components.css",
    
    # JS files
    "bika\static\bika\js\main.js",
    "bika\static\bika\js\cart.js",
    "bika\static\bika\js\checkout.js",
    "bika\static\bika\js\admin.js",
    "bika\static\bika\js\vendor.js",
    "bika\static\bika\js\ai_monitoring.js",
    
    # Image placeholders
    "bika\static\bika\images\logo.png",
    "bika\static\bika\images\favicon.ico",
    
    # Font files
    "bika\static\bika\fonts\bika-regular.woff",
    "bika\static\bika\fonts\bika-bold.woff2"
)

# Create all files
$fileCount = 0
foreach ($file in $templateFiles) {
    New-Item -ItemType File -Force -Path $file | Out-Null
    $fileCount++
    Write-Host "  Created: $file" -ForegroundColor Gray
}

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "Structure created successfully!" -ForegroundColor Green
Write-Host "Total directories created: $($directories.Count)" -ForegroundColor Cyan
Write-Host "Total files created: $fileCount" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Green

# Verify structure
Write-Host "`nVerifying structure..." -ForegroundColor Yellow
Get-ChildItem -Path "bika" -Recurse -Directory | Measure-Object | ForEach-Object { 
    Write-Host "Total directories: $($_.Count)" -ForegroundColor Gray 
}
Get-ChildItem -Path "bika" -Recurse -File | Measure-Object | ForEach-Object { 
    Write-Host "Total files: $($_.Count)" -ForegroundColor Gray 
}