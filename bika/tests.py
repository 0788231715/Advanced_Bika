from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.messages import get_messages
from bika.models import CustomUser, ProductCategory, Product, Cart
from bika.forms import ClientProductCreationForm, CustomerRegistrationForm
from decimal import Decimal
from django.core.files.uploadedfile import SimpleUploadedFile
import os
from django.conf import settings
import json

class ClientProductCreationFormTest(TestCase):
    def setUp(self):
        self.admin_user = CustomUser.objects.create_superuser(
            username='admin', email='admin@example.com', password='password'
        )
        self.customer_user = CustomUser.objects.create_user(
            username='customer1', email='customer1@example.com', password='password', user_type='customer'
        )
        self.category = ProductCategory.objects.create(name='Test Category')

    def test_form_valid_data(self):
        form_data = {
            'owner': self.customer_user.id,
            'name': 'Client Product A',
            'category': self.category.id,
            'description': 'Description for Client Product A',
            'price': '100.00',
            'storage_charges': '10.50',
            'client_price': '80.00',
            'stock_quantity': '10',
            'is_approved': True,
            'is_available': True,
        }
        form = ClientProductCreationForm(data=form_data)
        self.assertTrue(form.is_valid(), form.errors)

    def test_form_save_creates_product(self):
        form_data = {
            'owner': self.customer_user.id,
            'name': 'Client Product B',
            'category': self.category.id,
            'description': 'Description for Client Product B',
            'price': '150.00',
            'storage_charges': '12.00',
            'client_price': '110.00',
            'stock_quantity': '5',
            'is_approved': True,
            'is_available': True,
        }
        form = ClientProductCreationForm(data=form_data)
        self.assertTrue(form.is_valid())
        
        product = form.save(commit=False)
        product.vendor = self.admin_user # Assign a vendor
        product.save()

        self.assertEqual(Product.objects.count(), 1)
        new_product = Product.objects.first()
        self.assertEqual(new_product.name, 'Client Product B')
        self.assertEqual(new_product.owner, self.customer_user)
        self.assertEqual(new_product.vendor, self.admin_user)
        self.assertEqual(new_product.storage_charges, Decimal('12.00'))
        self.assertEqual(new_product.client_price, Decimal('110.00'))

    def test_form_invalid_negative_price(self):
        form_data = {
            'owner': self.customer_user.id,
            'name': 'Client Product C',
            'category': self.category.id,
            'description': 'Description for Client Product C',
            'price': '-10.00', # Invalid
            'storage_charges': '5.00',
            'client_price': '20.00',
            'stock_quantity': '1',
            'is_approved': True,
            'is_available': True,
        }
        form = ClientProductCreationForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('price', form.errors)

class CustomerRegistrationFormTest(TestCase):
    def test_form_valid_data(self):
        form_data = {
            'username': 'newcustomer',
            'email': 'new@example.com',
            'first_name': 'New',
            'last_name': 'Customer',
            'phone': '1234567890',
            'password': 'StrongPassword123!',  # Changed to stronger password
            'password2': 'StrongPassword123!', # Changed to stronger password
            'agree_terms': True,
            'user_type': 'customer', # Explicitly add user_type
        }
        form = CustomerRegistrationForm(data=form_data)
        self.assertTrue(form.is_valid(), form.errors)

    def test_form_save_creates_customer(self):
        form_data = {
            'username': 'anothercustomer',
            'email': 'another@example.com',
            'first_name': 'Another',
            'last_name': 'Customer',
            'phone': '0987654321',
            'password': 'EvenStrongerPassword456!', # Changed to stronger password
            'password2': 'EvenStrongerPassword456!',# Changed to stronger password
            'agree_terms': True,
            'user_type': 'customer', # Explicitly add user_type
        }
        form = CustomerRegistrationForm(data=form_data)
        self.assertTrue(form.is_valid())
        user = form.save()
        self.assertEqual(CustomUser.objects.count(), 1)
        self.assertEqual(user.username, 'anothercustomer')
        self.assertEqual(user.user_type, 'customer')

    def test_form_invalid_email_exists(self):
        CustomUser.objects.create_user(username='existing', email='exist@example.com', password='password')
        form_data = {
            'username': 'failcustomer',
            'email': 'exist@example.com', # Duplicate email
            'first_name': 'Fail',
            'last_name': 'Customer',
            'phone': '1111111111',
            'password': 'testpassword123!', # Stronger password
            'password2': 'wrongpassword123!', # Passwords mismatch
            'agree_terms': True,
            'user_type': 'customer', # Explicitly add user_type
        }
        form = CustomerRegistrationForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('email', form.errors)
        self.assertIn('password2', form.errors)


class RegisterClientViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin_user = CustomUser.objects.create_superuser(
            username='admin', email='admin@example.com', password='password', user_type='admin' # Explicitly set user_type
        )
        self.vendor_user = CustomUser.objects.create_user(
            username='vendor', email='vendor@example.com', password='password', user_type='vendor' # Explicitly set user_type
        )
        self.customer_user = CustomUser.objects.create_user(
            username='customer', email='customer@example.com', password='password', user_type='customer'
        )
        self.register_client_url = reverse('bika:register_client')

    def test_access_denied_for_unauthenticated_users(self):
        response = self.client.get(self.register_client_url)
        # Expected to redirect to /login/ with next param
        self.assertRedirects(response, reverse('bika:login') + '?next=' + self.register_client_url)

    def test_access_denied_for_customer_users(self):
        self.client.login(username='customer', password='password')
        response = self.client.get(self.register_client_url)
        # Expected to redirect to home with access denied message
        self.assertRedirects(response, reverse('bika:home'))
        messages = list(get_messages(response.wsgi_request))
        self.assertIn('Access denied.', str(messages[0]))

    def test_access_allowed_for_admin_users_get(self):
        self.client.login(username='admin', password='password')
        response = self.client.get(self.register_client_url)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context['form'], CustomerRegistrationForm)
        self.assertTemplateUsed(response, 'bika/pages/admin/register_client.html')

    def test_access_allowed_for_vendor_users_get(self):
        self.client.login(username='vendor', password='password')
        response = self.client.get(self.register_client_url)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context['form'], CustomerRegistrationForm)
        self.assertTemplateUsed(response, 'bika/pages/admin/register_client.html')

    def test_admin_registers_client_successfully(self):
        self.client.login(username='admin', password='password')
        form_data = {
            'username': 'newclient_by_admin',
            'email': 'newclient_admin@example.com',
            'first_name': 'New',
            'last_name': 'Client',
            'phone': '1112223333',
            'password': 'testpassword',
            'password2': 'testpassword',
            'agree_terms': True,
            'user_type': 'customer', # Explicitly add user_type
        }
        response = self.client.post(self.register_client_url, form_data, follow=True)
        # Assuming successful registration redirects to admin dashboard
        self.assertRedirects(response, reverse('bika:admin_dashboard'))
        self.assertTrue(CustomUser.objects.filter(username='newclient_by_admin', user_type='customer').exists())
        messages = list(get_messages(response.wsgi_request))
        self.assertIn('Client newclient_by_admin registered successfully!', str(messages[0]))

    def test_vendor_registers_client_successfully(self):
        self.client.login(username='vendor', password='password')
        form_data = {
            'username': 'newclient_by_vendor',
            'email': 'newclient_vendor@example.com',
            'first_name': 'Vendors',
            'last_name': 'Client',
            'phone': '4445556666',
            'password': 'testpassword',
            'password2': 'testpassword',
            'agree_terms': True,
            'user_type': 'customer', # Explicitly add user_type
        }
        response = self.client.post(self.register_client_url, form_data, follow=True)
        # Assuming successful registration redirects to admin dashboard
        self.assertRedirects(response, reverse('bika:admin_dashboard'))
        self.assertTrue(CustomUser.objects.filter(username='newclient_by_vendor', user_type='customer').exists())
        messages = list(get_messages(response.wsgi_request))
        self.assertIn('Client newclient_by_vendor registered successfully!', str(messages[0]))

    def test_admin_registers_client_with_invalid_data(self):
        self.client.login(username='admin', password='password')
        form_data = {
            'username': 'invalidclient',
            'email': 'invalid', # Invalid email
            'first_name': 'Invalid',
            'last_name': 'Data',
            'phone': '123',
            'password': 'testpassword',
            'password2': 'wrongpassword', # Passwords mismatch
            'agree_terms': False, # Not agreed
            'user_type': 'customer', # Explicitly add user_type
        }
        response = self.client.post(self.register_client_url, form_data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(CustomUser.objects.filter(username='invalidclient').exists())
        self.assertFormError(response, 'form', 'email', 'Enter a valid email address.')
        self.assertFormError(response, 'form', 'password2', 'The two password fields didn\'t match.')
        self.assertFormError(response, 'form', 'agree_terms', 'This field is required.')

class AddClientProductViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin_user = CustomUser.objects.create_superuser(
            username='admin', email='admin@example.com', password='password', user_type='admin' # Explicitly set user_type
        )
        self.vendor_user = CustomUser.objects.create_user(
            username='vendor', email='vendor@example.com', password='password', user_type='vendor' # Explicitly set user_type
        )
        self.customer_user = CustomUser.objects.create_user(
            username='customer1', email='customer1@example.com', password='password', user_type='customer'
        )
        self.category = ProductCategory.objects.create(name='Test Category')
        self.add_client_product_url = reverse('bika:add_client_product')
        self.dummy_image = SimpleUploadedFile("test_image.jpg", b"file_content", content_type="image/jpeg")

    def test_access_denied_for_unauthenticated_users(self):
        response = self.client.get(self.add_client_product_url)
        self.assertRedirects(response, reverse('bika:login') + '?next=' + self.add_client_product_url)

    def test_access_denied_for_customer_users(self):
        self.client.login(username='customer1', password='password')
        response = self.client.get(self.add_client_product_url)
        self.assertRedirects(response, reverse('bika:home'))
        messages = list(get_messages(response.wsgi_request))
        self.assertIn('Access denied.', str(messages[0]))

    def test_access_allowed_for_admin_users_get(self):
        self.client.login(username='admin', password='password')
        response = self.client.get(self.add_client_product_url)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context['form'], ClientProductCreationForm)
        self.assertTemplateUsed(response, 'bika/pages/admin/add_client_product.html')

    def test_access_allowed_for_vendor_users_get(self):
        self.client.login(username='vendor', password='password')
        response = self.client.get(self.add_client_product_url)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context['form'], ClientProductCreationForm)
        self.assertTemplateUsed(response, 'bika/pages/admin/add_client_product.html')

    def test_admin_adds_client_product_successfully(self):
        self.client.login(username='admin', password='password')
        form_data = {
            'owner': self.customer_user.id,
            'name': 'Admin Client Product',
            'category': self.category.id,
            'description': 'Description for Admin Client Product',
            'price': '200.00',
            'storage_charges': '15.00',
            'client_price': '180.00',
            'stock_quantity': '20',
            'is_approved': True,
            'is_available': True,
        }
        form_files = {'image': self.dummy_image}
        response = self.client.post(self.add_client_product_url, data=form_data, **form_files, follow=True)
        self.assertRedirects(response, reverse('bika:vendor_product_list')) # Redirects to vendor_product_list as per view
        
        self.assertEqual(Product.objects.count(), 1)
        new_product = Product.objects.first()
        self.assertEqual(new_product.name, 'Admin Client Product')
        self.assertEqual(new_product.owner, self.customer_user)
        self.assertEqual(new_product.vendor, self.admin_user)
        self.assertEqual(new_product.storage_charges, Decimal('15.00'))
        self.assertEqual(new_product.client_price, Decimal('180.00'))
        messages = list(get_messages(response.wsgi_request))
        self.assertIn(f'Product "Admin Client Product" added successfully for client {self.customer_user.username}!', str(messages[0]))

    def test_vendor_adds_client_product_successfully(self):
        self.client.login(username='vendor', password='password')
        form_data = {
            'owner': self.customer_user.id,
            'name': 'Vendor Client Product',
            'category': self.category.id,
            'description': 'Description for Vendor Client Product',
            'price': '250.00',
            'storage_charges': '18.00',
            'client_price': '220.00',
            'stock_quantity': '15',
            'is_approved': True,
            'is_available': True,
        }
        form_files = {'image': self.dummy_image}
        response = self.client.post(self.add_client_product_url, data=form_data, **form_files, follow=True)
        self.assertRedirects(response, reverse('bika:vendor_product_list'))
        
        self.assertEqual(Product.objects.count(), 1)
        new_product = Product.objects.first()
        self.assertEqual(new_product.name, 'Vendor Client Product')
        self.assertEqual(new_product.owner, self.customer_user)
        self.assertEqual(new_product.vendor, self.vendor_user)
        self.assertEqual(new_product.storage_charges, Decimal('18.00'))
        self.assertEqual(new_product.client_price, Decimal('220.00'))
        messages = list(get_messages(response.wsgi_request))
        self.assertIn(f'Product "Vendor Client Product" added successfully for client {self.customer_user.username}!', str(messages[0]))

    def test_admin_adds_client_product_with_invalid_data(self):
        self.client.login(username='admin', password='password')
        form_data = {
            'owner': self.customer_user.id,
            'name': '', # Invalid: Missing name
            'category': self.category.id,
            'description': 'Description',
            'price': '-10.00', # Invalid: Negative price
            'storage_charges': '5.00',
            'client_price': '20.00',
            'stock_quantity': '1',
            'is_approved': True,
            'is_available': True,
        }
        response = self.client.post(self.add_client_product_url, form_data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Product.objects.exists())
        self.assertFormError(response, 'form', 'name', 'This field is required.')
        self.assertFormError(response, 'form', 'price', 'Selling price cannot be negative.')

class CartFunctionalityTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = CustomUser.objects.create_user(
            username='testuser', email='test@example.com', password='password', user_type='customer'
        )
        self.category = ProductCategory.objects.create(name='Electronics')
        self.vendor = CustomUser.objects.create_user(
            username='vendor1', email='vendor@example.com', password='password', user_type='vendor'
        )
        self.product_in_stock = Product.objects.create(
            owner=self.user, vendor=self.vendor, category=self.category,
            name='Laptop', description='Powerful laptop', price=1200.00,
            stock_quantity=10, track_inventory=True, allow_backorders=False, slug='laptop',
            sku='LAPTOP001' # Unique SKU
        )
        self.product_out_of_stock = Product.objects.create(
            owner=self.user, vendor=self.vendor, category=self.category,
            name='Monitor', description='HD Monitor', price=300.00,
            stock_quantity=0, track_inventory=True, allow_backorders=False, slug='monitor',
            sku='MONITOR001' # Unique SKU
        )
        self.product_backorder_allowed = Product.objects.create(
            owner=self.user, vendor=self.vendor, category=self.category,
            name='Keyboard', description='Mechanical Keyboard', price=100.00,
            stock_quantity=2, track_inventory=True, allow_backorders=True, slug='keyboard',
            sku='KEYBOARD001' # Unique SKU
        )
        self.add_to_cart_url = reverse('bika:add_to_cart', args=[self.product_in_stock.id])
        self.quick_add_to_cart_url = reverse('bika:quick_add_to_cart', args=[self.product_in_stock.id])
        self.checkout_url = reverse('bika:checkout')
        self.client.login(username='testuser', password='password')

    def test_add_to_cart_sufficient_stock(self):
        initial_stock = self.product_in_stock.stock_quantity
        response = self.client.post(self.add_to_cart_url, {'quantity': 1}, follow=True)
        # Expected to redirect to product_detail
        self.assertRedirects(response, reverse('bika:product_detail', args=[self.product_in_stock.slug]))
        self.assertEqual(Cart.objects.filter(user=self.user, product=self.product_in_stock).first().quantity, 1)
        self.product_in_stock.refresh_from_db()
        self.assertEqual(self.product_in_stock.stock_quantity, initial_stock - 1)
        messages = list(get_messages(response.wsgi_request))
        self.assertIn(f'1 x "{self.product_in_stock.name}" added to cart!', str(messages[0]))

    def test_add_to_cart_insufficient_stock_no_backorder(self):
        response = self.client.post(self.add_to_cart_url, {'quantity': 11}, follow=True)
        # Expected to redirect to product_detail
        self.assertRedirects(response, reverse('bika:product_detail', args=[self.product_in_stock.slug]))
        self.assertFalse(Cart.objects.filter(user=self.user, product=self.product_in_stock).exists())
        messages = list(get_messages(response.wsgi_request))
        self.assertIn(f'Only {self.product_in_stock.stock_quantity} of "{self.product_in_stock.name}" are available in stock.', str(messages[0]))
        self.product_in_stock.refresh_from_db()
        self.assertEqual(self.product_in_stock.stock_quantity, 10) # Stock should not change

    def test_quick_add_to_cart_sufficient_stock(self):
        initial_stock = self.product_in_stock.stock_quantity
        response = self.client.post(self.quick_add_to_cart_url, {'quantity': 1}, xhr=True)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['success'])
        self.assertEqual(Cart.objects.filter(user=self.user, product=self.product_in_stock).first().quantity, 1)
        self.product_in_stock.refresh_from_db()
        self.assertEqual(self.product_in_stock.stock_quantity, initial_stock - 1)
        self.assertEqual(data['new_stock_quantity'], initial_stock - 1)
        self.assertFalse(data['allow_backorders'])

    def test_quick_add_to_cart_insufficient_stock_no_backorder(self):
        response = self.client.post(self.quick_add_to_cart_url, {'quantity': 11}, xhr=True)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertFalse(data['success'])
        self.assertFalse(Cart.objects.filter(user=self.user, product=self.product_in_stock).exists())
        self.product_in_stock.refresh_from_db()
        self.assertEqual(self.product_in_stock.stock_quantity, 10) # Stock should not change
        self.assertEqual(data['new_stock_quantity'], 10)
        self.assertFalse(data['allow_backorders'])

    def test_quick_add_to_cart_backorder_allowed(self):
        initial_stock = self.product_backorder_allowed.stock_quantity
        url = reverse('bika:quick_add_to_cart', args=[self.product_backorder_allowed.id])
        response = self.client.post(url, {'quantity': 5}, xhr=True) # Order more than stock
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['success'])
        self.assertEqual(Cart.objects.filter(user=self.user, product=self.product_backorder_allowed).first().quantity, 5)
        self.product_backorder_allowed.refresh_from_db()
        # Stock should not change if backorder allowed and inventory tracked
        self.assertEqual(self.product_backorder_allowed.stock_quantity, initial_stock) 
        self.assertEqual(data['new_stock_quantity'], initial_stock)
        self.assertTrue(data['allow_backorders'])

    def test_buy_now_functionality(self):
        # This tests the JS client-side behavior, but we can simulate the POST to add_to_cart and check redirect
        # Assuming buyNow triggers quick_add_to_cart and then redirects
        # Since buyNow is client-side, we test the resulting server-side actions.
        # Here we test quick_add_to_cart, and then verify the preconditions.
        # The product should be in the cart.
        response = self.client.post(self.quick_add_to_cart_url, {'quantity': 1}, xhr=True)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['success'])
        
        # After success, the JS would redirect to checkout.
        # We can't directly test the JS redirect here, but we verify the preconditions.
        # The product should be in the cart.
        self.assertTrue(Cart.objects.filter(user=self.user, product=self.product_in_stock).exists())