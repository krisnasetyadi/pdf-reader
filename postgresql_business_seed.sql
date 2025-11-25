-- =====================================================
-- PostgreSQL Data Seed untuk Sistem Bisnis
-- File: postgresql_business_seed.sql
-- =====================================================

-- =====================================================
-- 1. SEED DATA - COMPANY PROFILES
-- =====================================================
INSERT INTO company_profiles (company_name, company_code, industry, founded_year, headquarters, phone, email, website, description, annual_revenue, employee_count, stock_symbol, is_public) VALUES
('PT. Teknologi Nusantara', 'TEKNUS', 'Technology', 2020, 'Jakarta Selatan', '021-12345678', 'info@teknusantara.com', 'www.teknusantara.com', 'Perusahaan teknologi yang fokus pada solusi AI dan Machine Learning untuk bisnis di Indonesia', 50000000000, 150, 'TEKN', TRUE),
('PT. Bank Central Asia Tbk', 'BCA', 'Banking', 1957, 'Jakarta Pusat', '021-2358-8000', 'info@bca.co.id', 'www.bca.co.id', 'Bank swasta terbesar di Indonesia dengan jaringan cabang yang luas', 89000000000000, 25000, 'BBCA', TRUE),
('PT. Telkom Indonesia Tbk', 'TLKM', 'Telecommunications', 1965, 'Bandung', '022-2500-1234', 'info@telkom.co.id', 'www.telkom.co.id', 'BUMN yang bergerak di bidang telekomunikasi dan teknologi informasi', 134000000000000, 24000, 'TLKM', TRUE),
('CV. Maju Bersama', 'MABER', 'Retail', 2015, 'Surabaya', '031-7654321', 'info@majubersama.com', 'www.majubersama.com', 'Toko retail yang menjual berbagai keperluan rumah tangga dan elektronik', 2500000000, 25, NULL, FALSE),
('PT. Indofood Sukses Makmur', 'INDF', 'Food & Beverage', 1990, 'Jakarta Utara', '021-6505-8888', 'info@indofood.com', 'www.indofood.com', 'Produsen makanan dan minuman terbesar di Indonesia', 78000000000000, 85000, 'INDF', TRUE);

-- =====================================================
-- 2. SEED DATA - CATEGORIES
-- =====================================================
INSERT INTO categories (category_code, category_name, description) VALUES
('ELEC', 'Electronics', 'Peralatan elektronik dan gadget'),
('FOOD', 'Food & Beverage', 'Makanan dan minuman'),
('CLOTH', 'Clothing', 'Pakaian dan fashion'),
('BOOK', 'Books', 'Buku dan alat tulis'),
('HOME', 'Home Appliances', 'Peralatan rumah tangga'),
('AUTO', 'Automotive', 'Otomotif dan spare parts'),
('HEALTH', 'Health & Beauty', 'Kesehatan dan kecantikan'),
('SPORT', 'Sports', 'Olahraga dan outdoor');

-- Sub-categories
INSERT INTO categories (category_code, category_name, parent_category_id, description) VALUES
('ELEC-MOB', 'Mobile Phones', 1, 'Smartphone dan aksesoris'),
('ELEC-LAP', 'Laptops', 1, 'Laptop dan komputer'),
('FOOD-SNK', 'Snacks', 2, 'Makanan ringan dan cemilan'),
('FOOD-BEV', 'Beverages', 2, 'Minuman dan soft drink'),
('CLOTH-MEN', 'Mens Clothing', 3, 'Pakaian pria'),
('CLOTH-WOM', 'Womens Clothing', 3, 'Pakaian wanita');

-- =====================================================
-- 3. SEED DATA - EMPLOYEES
-- =====================================================
INSERT INTO employee_profiles (employee_id, first_name, last_name, email, phone, department, position, salary, hire_date, birth_date, address, company_id) VALUES
('EMP001', 'Budi', 'Santoso', 'budi.santoso@teknusantara.com', '081234567890', 'Sales', 'Sales Manager', 15000000, '2021-01-15', '1985-03-20', 'Jl. Sudirman No. 123, Jakarta Selatan', 1),
('EMP002', 'Sari', 'Dewi', 'sari.dewi@teknusantara.com', '081234567891', 'Sales', 'Sales Executive', 8000000, '2021-06-01', '1990-07-15', 'Jl. Kemang Raya No. 45, Jakarta Selatan', 1),
('EMP003', 'Ahmad', 'Rahman', 'ahmad.rahman@majubersama.com', '081234567892', 'Sales', 'Store Manager', 12000000, '2020-03-10', '1988-12-05', 'Jl. Raya Gubeng No. 67, Surabaya', 4),
('EMP004', 'Rina', 'Wijaya', 'rina.wijaya@majubersama.com', '081234567893', 'Sales', 'Cashier', 5000000, '2021-08-20', '1995-05-30', 'Jl. Dharmawangsa No. 89, Surabaya', 4),
('EMP005', 'Dedi', 'Kurniawan', 'dedi.kurniawan@teknusantara.com', '081234567894', 'IT', 'Software Engineer', 18000000, '2020-11-01', '1987-09-12', 'Jl. Senopati No. 12, Jakarta Selatan', 1);

-- Set manager relationships
UPDATE employee_profiles SET manager_id = 1 WHERE id = 2; -- Sari reports to Budi
UPDATE employee_profiles SET manager_id = 3 WHERE id = 4; -- Rina reports to Ahmad

-- =====================================================
-- 4. SEED DATA - CUSTOMERS
-- =====================================================
INSERT INTO customers (customer_code, customer_name, customer_type, email, phone, address, city, province, postal_code, credit_limit, payment_terms) VALUES
('CUST001', 'PT. Mitra Sejahtera', 'corporate', 'admin@mitrasejahtera.com', '021-5555-1234', 'Jl. Gatot Subroto No. 100', 'Jakarta', 'DKI Jakarta', '12950', 50000000, 30),
('CUST002', 'Toko Elektronik Jaya', 'corporate', 'info@elektronikjaya.com', '031-7777-5678', 'Jl. Basuki Rahmat No. 25', 'Surabaya', 'Jawa Timur', '60271', 25000000, 21),
('CUST003', 'John Doe', 'individual', 'john.doe@email.com', '081234567895', 'Jl. Menteng Raya No. 45', 'Jakarta', 'DKI Jakarta', '10310', 5000000, 0),
('CUST004', 'Siti Aminah', 'individual', 'siti.aminah@email.com', '081234567896', 'Jl. Malioboro No. 78', 'Yogyakarta', 'DI Yogyakarta', '55213', 3000000, 0),
('CUST005', 'CV. Berkah Jaya', 'corporate', 'info@berkahjaya.com', '022-8888-9999', 'Jl. Cihampelas No. 150', 'Bandung', 'Jawa Barat', '40131', 15000000, 14),
('CUST006', 'Warung Pak Budi', 'individual', 'pakbudi@warung.com', '081234567897', 'Jl. Malang Raya No. 33', 'Malang', 'Jawa Timur', '65145', 2000000, 7),
('CUST007', 'PT. Digital Solutions', 'corporate', 'contact@digitalsol.com', '021-4444-7777', 'Jl. Kuningan No. 88', 'Jakarta', 'DKI Jakarta', '12940', 75000000, 30);

-- =====================================================
-- 5. SEED DATA - PRODUCTS
-- =====================================================
INSERT INTO products (product_code, product_name, category_id, description, unit_price, cost_price, stock_quantity, minimum_stock, unit, weight, brand, supplier) VALUES
-- Electronics
('PROD001', 'Samsung Galaxy S23', 9, 'Smartphone flagship Samsung dengan kamera 200MP', 15000000, 12000000, 25, 5, 'pcs', 0.168, 'Samsung', 'PT. Samsung Electronics Indonesia'),
('PROD002', 'iPhone 14 Pro', 9, 'Smartphone premium Apple dengan chip A16 Bionic', 20000000, 16000000, 15, 3, 'pcs', 0.206, 'Apple', 'PT. Apple Indonesia'),
('PROD003', 'MacBook Air M2', 10, 'Laptop ultrabook Apple dengan chip M2', 18000000, 14500000, 10, 2, 'pcs', 1.24, 'Apple', 'PT. Apple Indonesia'),
('PROD004', 'Lenovo ThinkPad X1', 10, 'Business laptop dengan performa tinggi', 25000000, 20000000, 8, 2, 'pcs', 1.13, 'Lenovo', 'PT. Lenovo Indonesia'),

-- Food & Beverage
('PROD005', 'Indomie Goreng', 11, 'Mi instan rasa rendang, kemasan 5 bungkus', 12000, 8000, 500, 100, 'pack', 0.425, 'Indomie', 'PT. Indofood CBP'),
('PROD006', 'Teh Botol Sosro', 12, 'Minuman teh dalam kemasan botol 450ml', 3500, 2500, 200, 50, 'botol', 0.5, 'Sosro', 'PT. Sinar Sosro'),
('PROD007', 'Chitato Rasa Sapi Panggang', 11, 'Keripik kentang rasa sapi panggang 68gr', 8000, 6000, 150, 30, 'pcs', 0.068, 'Chitato', 'PT. Indofood Fritolay'),
('PROD008', 'Aqua Botol 600ml', 12, 'Air mineral dalam kemasan botol', 2500, 1800, 300, 80, 'botol', 0.6, 'Aqua', 'PT. Aqua Golden Mississippi'),

-- Clothing
('PROD009', 'Kemeja Batik Pria', 13, 'Kemeja batik motif parang ukuran L', 250000, 180000, 30, 5, 'pcs', 0.3, 'Batik Trusmi', 'CV. Batik Nusantara'),
('PROD010', 'Dress Wanita Casual', 14, 'Dress wanita untuk acara casual, bahan cotton', 180000, 120000, 20, 5, 'pcs', 0.25, 'Local Brand', 'CV. Fashion Indonesia'),

-- Home Appliances  
('PROD011', 'Rice Cooker Miyako', 5, 'Rice cooker kapasitas 1.8 liter', 450000, 350000, 25, 5, 'pcs', 2.5, 'Miyako', 'PT. Miyako Electronics'),
('PROD012', 'Blender Philips', 5, 'Blender 2 liter dengan 5 kecepatan', 800000, 650000, 15, 3, 'pcs', 3.2, 'Philips', 'PT. Philips Indonesia');

-- =====================================================
-- 6. SEED DATA - TRANSACTIONS
-- =====================================================
INSERT INTO transactions (transaction_no, transaction_date, customer_id, employee_id, transaction_type, payment_method, payment_status, subtotal, tax_amount, discount_amount, total_amount, due_date) VALUES
('TXN-20241101-001', '2024-11-01', 1, 1, 'sale', 'bank_transfer', 'paid', 45000000, 4500000, 2000000, 47500000, '2024-12-01'),
('TXN-20241102-002', '2024-11-02', 3, 2, 'sale', 'cash', 'paid', 20000000, 2000000, 0, 22000000, NULL),
('TXN-20241103-003', '2024-11-03', 2, 3, 'sale', 'credit', 'pending', 15000000, 1500000, 500000, 16000000, '2024-11-24'),
('TXN-20241104-004', '2024-11-04', 4, 4, 'sale', 'cash', 'paid', 250000, 25000, 0, 275000, NULL),
('TXN-20241105-005', '2024-11-05', 5, 1, 'sale', 'credit_card', 'paid', 18000000, 1800000, 1000000, 18800000, NULL),
('TXN-20241106-006', '2024-11-06', 6, 4, 'sale', 'cash', 'paid', 50000, 5000, 5000, 50000, NULL),
('TXN-20241107-007', '2024-11-07', 7, 2, 'sale', 'bank_transfer', 'pending', 25000000, 2500000, 0, 27500000, '2024-12-07');

-- =====================================================
-- 7. SEED DATA - TRANSACTION DETAILS
-- =====================================================
INSERT INTO transaction_details (transaction_id, product_id, quantity, unit_price, discount_percent, discount_amount, line_total) VALUES
-- Transaction 1: Corporate customer buying laptops and phones
(1, 3, 1, 18000000, 0, 0, 18000000), -- MacBook Air M2
(1, 4, 1, 25000000, 5, 1250000, 23750000), -- Lenovo ThinkPad X1 with discount
(1, 1, 1, 15000000, 5, 750000, 14250000), -- Samsung Galaxy S23 with discount

-- Transaction 2: Individual buying iPhone
(2, 2, 1, 20000000, 0, 0, 20000000), -- iPhone 14 Pro

-- Transaction 3: Electronics store buying phones
(3, 1, 1, 15000000, 0, 0, 15000000), -- Samsung Galaxy S23

-- Transaction 4: Individual buying clothing
(4, 9, 1, 250000, 0, 0, 250000), -- Kemeja Batik Pria

-- Transaction 5: Corporate buying laptop
(5, 3, 1, 18000000, 5, 900000, 17100000), -- MacBook Air M2 with discount

-- Transaction 6: Small purchase - snacks
(6, 5, 2, 12000, 0, 0, 24000), -- Indomie Goreng
(6, 6, 5, 3500, 0, 0, 17500), -- Teh Botol Sosro
(6, 7, 1, 8000, 10, 800, 7200), -- Chitato with discount

-- Transaction 7: Corporate buying ThinkPad
(7, 4, 1, 25000000, 0, 0, 25000000); -- Lenovo ThinkPad X1

-- Update stock quantities after sales
UPDATE products SET stock_quantity = stock_quantity - 2 WHERE id = 1; -- Samsung Galaxy S23: 2 sold
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = 2; -- iPhone 14 Pro: 1 sold  
UPDATE products SET stock_quantity = stock_quantity - 2 WHERE id = 3; -- MacBook Air M2: 2 sold
UPDATE products SET stock_quantity = stock_quantity - 2 WHERE id = 4; -- Lenovo ThinkPad X1: 2 sold
UPDATE products SET stock_quantity = stock_quantity - 2 WHERE id = 5; -- Indomie: 2 packs sold
UPDATE products SET stock_quantity = stock_quantity - 5 WHERE id = 6; -- Teh Botol: 5 sold
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = 7; -- Chitato: 1 sold
UPDATE products SET stock_quantity = stock_quantity - 1 WHERE id = 9; -- Kemeja Batik: 1 sold

-- =====================================================
-- 8. ADDITIONAL SAMPLE QUERIES FOR TESTING
-- =====================================================

-- Query untuk melihat total penjualan per bulan
-- SELECT 
--     EXTRACT(YEAR FROM transaction_date) AS year,
--     EXTRACT(MONTH FROM transaction_date) AS month,
--     SUM(total_amount) AS monthly_sales,
--     COUNT(*) AS total_transactions
-- FROM transactions 
-- WHERE transaction_type = 'sale' 
-- GROUP BY EXTRACT(YEAR FROM transaction_date), EXTRACT(MONTH FROM transaction_date)
-- ORDER BY year, month;

-- Query untuk melihat produk terlaris
-- SELECT 
--     p.product_name,
--     SUM(td.quantity) AS total_sold,
--     SUM(td.line_total) AS total_revenue
-- FROM transaction_details td
-- JOIN products p ON td.product_id = p.id
-- JOIN transactions t ON td.transaction_id = t.id
-- WHERE t.transaction_type = 'sale'
-- GROUP BY p.id, p.product_name
-- ORDER BY total_sold DESC;

-- Query untuk melihat performa sales per employee
-- SELECT 
--     e.first_name || ' ' || e.last_name AS employee_name,
--     COUNT(t.id) AS total_transactions,
--     SUM(t.total_amount) AS total_sales
-- FROM employee_profiles e
-- LEFT JOIN transactions t ON e.id = t.employee_id AND t.transaction_type = 'sale'
-- GROUP BY e.id, e.first_name, e.last_name
-- ORDER BY total_sales DESC;