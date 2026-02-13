# bika/recommendation_service.py
from django.db.models import Count, F
from django.utils import timezone
from bika.models import Product, ProductCategory, OrderItem, CustomUser, ProductReview, Cart, Wishlist
from datetime import timedelta

class ProductRecommendationService:
    """
    Service for generating various types of product recommendations for customers.
    Initially focuses on rule-based recommendations, with a pathway for future AI integration.
    """

    def get_related_products(self, product, limit=4):
        """
        Recommends products related to a given product.
        Currently based on category. Can be extended with AI-driven similarity.
        """
        if not product:
            return Product.objects.none()

        # Start with products from the same primary category
        related_products = Product.objects.filter(
            category=product.category,
            status='active'
        ).exclude(id=product.id)

        # Enhance: If subcategories exist, also pull from parent or sibling categories
        if product.category.parent:
            # Add products from the parent category
            related_products = (related_products | Product.objects.filter(
                category=product.category.parent,
                status='active'
            ).exclude(id=product.id)).distinct()

            # Add products from sibling categories
            sibling_categories = ProductCategory.objects.filter(
                parent=product.category.parent
            ).exclude(id=product.category.id)
            for sibling_cat in sibling_categories:
                related_products = (related_products | Product.objects.filter(
                    category=sibling_cat,
                    status='active'
                ).exclude(id=product.id)).distinct()

        # Further enhance: basic tag-based similarity
        if product.tags:
            product_tags = [tag.strip() for tag in product.tags.split(',') if tag.strip()]
            if product_tags:
                for tag in product_tags:
                    related_products = (related_products | Product.objects.filter(
                        tags__icontains=tag,
                        status='active'
                    ).exclude(id=product.id)).distinct()

        return related_products.order_by('?')[:limit] # Order randomly for variety


    def get_popular_products(self, limit=8, time_period_days=30):
        """
        Recommends popular products based on sales, views, or cart additions.
        """
        # 1. Popular by sales (most ordered items in a given period)
        sales_popular_products_ids = OrderItem.objects.filter(
            order__created_at__gte=timezone.now() - timedelta(days=time_period_days),
            order__status__in=['confirmed', 'shipped', 'delivered']
        ).values('product').annotate(
            total_quantity=Count('quantity')
        ).order_by('-total_quantity').values_list('product_id', flat=True)[:limit]

        sales_products = Product.objects.filter(id__in=list(sales_popular_products_ids), status='active')

        # 2. Popular by views (most viewed products)
        viewed_popular_products = Product.objects.filter(
            status='active'
        ).order_by('-views_count')[:limit]

        # 3. Popular by cart additions (items most frequently added to carts)
        cart_popular_products_ids = Cart.objects.filter(
            added_at__gte=timezone.now() - timedelta(days=time_period_days)
        ).values('product').annotate(
            cart_count=Count('product')
        ).order_by('-cart_count').values_list('product_id', flat=True)[:limit]

        cart_products = Product.objects.filter(id__in=list(cart_popular_products_ids), status='active')

        # Combine and deduplicate, prioritizing sales, then views, then cart additions
        combined_products_ids = list(sales_products.values_list('id', flat=True))
        combined_products_ids.extend(list(viewed_popular_products.values_list('id', flat=True)))
        combined_products_ids.extend(list(cart_products.values_list('id', flat=True)))

        # Get unique product IDs while preserving approximate order of importance
        unique_products_ids = []
        seen_ids = set()
        for p_id in combined_products_ids:
            if p_id not in seen_ids:
                unique_products_ids.append(p_id)
                seen_ids.add(p_id)

        # Fetch products based on unique IDs
        # To maintain order, use a custom sort or fetch individually if limit is small
        products_map = {p.id: p for p in Product.objects.filter(id__in=unique_products_ids, status='active')}
        ordered_products = [products_map[p_id] for p_id in unique_products_ids if p_id in products_map]

        return ordered_products[:limit]


    def get_personalized_recommendations(self, user: CustomUser, limit=8, time_period_days=90):
        """
        Generates personalized product recommendations for a specific user.
        This is a placeholder for more advanced AI/ML models.
        """
        if not user.is_authenticated:
            return self.get_popular_products(limit=limit) # Fallback for anonymous users

        # Placeholder logic:
        # 1. Products from categories the user has previously purchased
        user_purchased_product_ids = OrderItem.objects.filter(
            order__user=user,
            order__created_at__gte=timezone.now() - timedelta(days=time_period_days)
        ).values_list('product_id', flat=True).distinct()

        user_purchased_categories_ids = Product.objects.filter(
            id__in=user_purchased_product_ids
        ).values_list('category_id', flat=True).distinct()

        category_recommendations = Product.objects.filter(
            category__in=user_purchased_categories_ids,
            status='active'
        ).exclude(id__in=user_purchased_product_ids).order_by('?')[:limit]

        # 2. Products from user's wishlist (if any, not really recommendations but important for personalization)
        wishlist_products = Product.objects.filter(
            wishlist__user=user,
            status='active'
        )

        # 3. Combine and deduplicate
        combined_recommendations = list(category_recommendations)
        combined_recommendations.extend(list(wishlist_products)) # Wishlist items are high-priority recommendations

        unique_recommendations = []
        seen_ids = set()
        for product in combined_recommendations:
            if product.id not in seen_ids:
                unique_recommendations.append(product)
                seen_ids.add(product.id)
        
        # Ensure we don't return more than limit
        if len(unique_recommendations) < limit:
            # Fill with popular products if not enough personalized ones
            popular_fillers = self.get_popular_products(limit=limit * 2) # Get more to filter
            for product in popular_fillers:
                if product.id not in seen_ids:
                    unique_recommendations.append(product)
                    seen_ids.add(product.id)
                if len(unique_recommendations) >= limit:
                    break

        return unique_recommendations[:limit]

    def get_cross_sell_recommendations(self, cart_products_or_order_items, limit=4):
        """
        Recommends products that are frequently bought together (cross-sell).
        Requires analyzing historical order data.
        """
        if not cart_products_or_order_items:
            return Product.objects.none()

        # Get product IDs from current cart or order
        current_product_ids = [item.product.id for item in cart_products_or_order_items]

        # Find other products frequently bought with these products
        # This is a simplified approach, a true "bought together" requires more complex logic
        # For simplicity, we find products in orders that also contain any of the current products
        
        # Get orders that contain any of the current products
        relevant_orders = OrderItem.objects.filter(
            product_id__in=current_product_ids,
            order__status__in=['confirmed', 'shipped', 'delivered']
        ).values_list('order_id', flat=True)

        # Get all items from those relevant orders, excluding the current products
        cross_sell_product_ids = OrderItem.objects.filter(
            order_id__in=list(relevant_orders)
        ).exclude(
            product_id__in=current_product_ids
        ).values('product').annotate(
            frequency=Count('product')
        ).order_by('-frequency').values_list('product_id', flat=True)[:limit * 2] # Get more to filter

        return Product.objects.filter(id__in=list(cross_sell_product_ids), status='active').order_by('?')[:limit]

    def get_upsell_recommendations(self, product, limit=4):
        """
        Recommends higher-value or premium alternatives to a given product (upsell).
        Currently based on same category and higher price. Can be extended with feature similarity.
        """
        if not product:
            return Product.objects.none()
        
        upsell_products = Product.objects.filter(
            category=product.category,
            price__gt=product.price, # Higher price
            status='active'
        ).exclude(id=product.id).order_by('-price')[:limit] # Prioritize higher price

        # If not enough, consider products with better ratings (placeholder logic)
        if upsell_products.count() < limit:
            better_rated_products = Product.objects.filter(
                category=product.category,
                status='active',
                # This needs average rating field on Product or a subquery
                # For now, let's just grab more products from the same category
            ).exclude(id__in=list(upsell_products.values_list('id', flat=True)) + [product.id]).order_by('?')[:limit - upsell_products.count()]
            upsell_products = list(upsell_products) + list(better_rated_products)

        return upsell_products[:limit]
