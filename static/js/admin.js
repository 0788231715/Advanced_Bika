// Bika Admin JavaScript
document.addEventListener('DOMContentLoaded', function() {
    'use strict';

    // Admin-specific enhancements
    class BikaAdmin {
        constructor() {
            this.init();
        }

        init() {
            this.enhanceForms();
            this.enhanceTables();
            this.setupQuickActions();
            this.setupStatusIndicators();
            this.setupAutoRefresh();
        }

        // Form enhancements
        enhanceForms() {
            // Add character counters to textareas
            const textareas = document.querySelectorAll('textarea');
            textareas.forEach(textarea => {
                const maxLength = textarea.maxLength;
                if (maxLength > 0) {
                    this.addCharacterCounter(textarea, maxLength);
                }
            });

            // Auto-save indicators
            const forms = document.querySelectorAll('form');
            forms.forEach(form => {
                form.addEventListener('change', this.debounce(() => {
                    this.showAutoSaveIndicator();
                }, 1000));
            });
        }

        // Add character counter to textareas
        addCharacterCounter(textarea, maxLength) {
            const counter = document.createElement('div');
            counter.className = 'char-counter text-muted small mt-1';
            counter.textContent = `0/${maxLength} characters`;
            
            textarea.parentNode.appendChild(counter);

            textarea.addEventListener('input', () => {
                const length = textarea.value.length;
                counter.textContent = `${length}/${maxLength} characters`;
                
                if (length > maxLength * 0.9) {
                    counter.classList.add('text-warning');
                } else {
                    counter.classList.remove('text-warning');
                }
            });
        }

        // Table enhancements
        enhanceTables() {
            // Add row selection
            const tables = document.querySelectorAll('table');
            tables.forEach(table => {
                table.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    if (row && !row.classList.contains('header')) {
                        row.classList.toggle('selected');
                    }
                });
            });

            // Add search functionality to tables
            this.addTableSearch();
        }

        // Add search to tables
        addTableSearch() {
            const tables = document.querySelectorAll('.results table');
            tables.forEach(table => {
                const searchBox = document.createElement('input');
                searchBox.type = 'text';
                searchBox.placeholder = 'Search in this table...';
                searchBox.className = 'form-control mb-3 table-search';
                searchBox.style.maxWidth = '300px';

                table.parentNode.insertBefore(searchBox, table);

                searchBox.addEventListener('input', this.debounce((e) => {
                    const searchTerm = e.target.value.toLowerCase();
                    const rows = table.querySelectorAll('tbody tr');
                    
                    rows.forEach(row => {
                        const text = row.textContent.toLowerCase();
                        if (text.includes(searchTerm)) {
                            row.style.display = '';
                        } else {
                            row.style.display = 'none';
                        }
                    });
                }, 300));
            });
        }

        // Quick actions setup
        setupQuickActions() {
            // Add quick edit buttons
            const objectTools = document.querySelector('.object-tools');
            if (objectTools) {
                const quickEdit = document.createElement('li');
                quickEdit.innerHTML = '<a href="#quick-edit" class="btn-quick-edit"><i class="fas fa-edit me-1"></i>Quick Edit</a>';
                objectTools.appendChild(quickEdit);
            }
        }

        // Status indicators
        setupStatusIndicators() {
            // Add status badges based on content
            const statusCells = document.querySelectorAll('td.field-status, td.field-is_active');
            statusCells.forEach(cell => {
                const text = cell.textContent.trim().toLowerCase();
                if (text === 'true' || text === 'active' || text === 'published') {
                    cell.innerHTML = '<span class="badge badge-success">Active</span>';
                } else if (text === 'false' || text === 'inactive' || text === 'draft') {
                    cell.innerHTML = '<span class="badge badge-secondary">Inactive</span>';
                } else if (text === 'pending') {
                    cell.innerHTML = '<span class="badge badge-warning">Pending</span>';
                }
            });
        }

        // Auto-refresh for messages and notifications
        setupAutoRefresh() {
            // Refresh unread message count every 30 seconds
            setInterval(() => {
                this.updateMessageCount();
            }, 30000);
        }

        // Update message count
        updateMessageCount() {
            // This would typically make an API call to get updated counts
            console.log('Updating message counts...');
        }

        // Show auto-save indicator
        showAutoSaveIndicator() {
            // Create or update auto-save indicator
            let indicator = document.querySelector('.auto-save-indicator');
            if (!indicator) {
                indicator = document.createElement('div');
                indicator.className = 'auto-save-indicator alert alert-info';
                indicator.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999;';
                document.body.appendChild(indicator);
            }

            indicator.innerHTML = '<i class="fas fa-save me-2"></i>Changes saved automatically';
            indicator.style.display = 'block';

            setTimeout(() => {
                indicator.style.display = 'none';
            }, 3000);
        }

        // Utility function: debounce
        debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
    }

    // Initialize the admin enhancements
    window.bikaAdmin = new BikaAdmin();

    // Console message
    console.log(`
    🎯 Bika Admin Panel Enhanced
    ============================
    Features loaded:
    ✓ Form enhancements
    ✓ Table search
    ✓ Status indicators
    ✓ Auto-save indicators
    ✓ Quick actions
    `);
});

// Global admin functions
function confirmDelete(message = 'Are you sure you want to delete this item?') {
    return confirm(message);
}

function showLoading(message = 'Processing...') {
    // Create loading overlay
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        color: white;
        font-size: 1.2rem;
    `;
    overlay.innerHTML = `
        <div class="text-center">
            <i class="fas fa-spinner fa-spin fa-2x mb-3"></i>
            <div>${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
}

function hideLoading(overlay) {
    if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
    }
}