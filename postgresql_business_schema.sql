-- =====================================================
-- PostgreSQL Database Schema untuk Data Bisnis
-- File: postgresql_business_schema.sql
-- =====================================================

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS transaction_details CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS employee_profiles CASCADE;
DROP TABLE IF EXISTS company_profiles CASCADE;
DROP TABLE IF EXISTS categories CASCADE;

-- =====================================================
-- 1. COMPANY PROFILES TABLE
-- =====================================================
CREATE TABLE company_profiles (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    company_code VARCHAR(10) UNIQUE NOT NULL,
    industry VARCHAR(100),
    founded_year INTEGER,
    headquarters VARCHAR(255),
    phone VARCHAR(20),
    email VARCHAR(100),
    website VARCHAR(255),
    description TEXT,
    annual_revenue DECIMAL(15,2),
    employee_count INTEGER,
    stock_symbol VARCHAR(10),
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- 2. EMPLOYEE PROFILES TABLE
-- =====================================================
CREATE TABLE employee_profiles (
    id SERIAL PRIMARY KEY,
    employee_id VARCHAR(20) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    department VARCHAR(100),
    position VARCHAR(100),
    salary DECIMAL(12,2),
    hire_date DATE,
    birth_date DATE,
    address TEXT,
    manager_id INTEGER REFERENCES employee_profiles(id),
    company_id INTEGER REFERENCES company_profiles(id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- 3. CUSTOMERS TABLE
-- =====================================================
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    customer_code VARCHAR(20) UNIQUE NOT NULL,
    customer_name VARCHAR(255) NOT NULL,
    customer_type VARCHAR(50) CHECK (customer_type IN ('individual', 'corporate')),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    province VARCHAR(100),
    postal_code VARCHAR(10),
    country VARCHAR(100) DEFAULT 'Indonesia',
    tax_id VARCHAR(30),
    credit_limit DECIMAL(15,2),
    payment_terms INTEGER DEFAULT 30, -- days
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- 4. CATEGORIES TABLE
-- =====================================================
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    category_code VARCHAR(20) UNIQUE NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    parent_category_id INTEGER REFERENCES categories(id),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- 5. PRODUCTS TABLE
-- =====================================================
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_code VARCHAR(30) UNIQUE NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    category_id INTEGER REFERENCES categories(id),
    description TEXT,
    unit_price DECIMAL(12,2) NOT NULL,
    cost_price DECIMAL(12,2),
    stock_quantity INTEGER DEFAULT 0,
    minimum_stock INTEGER DEFAULT 0,
    unit VARCHAR(20) DEFAULT 'pcs',
    weight DECIMAL(8,2), -- kg
    dimensions VARCHAR(50), -- e.g., "10x15x5 cm"
    brand VARCHAR(100),
    supplier VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- 6. TRANSACTIONS TABLE (Sales)
-- =====================================================
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_no VARCHAR(30) UNIQUE NOT NULL,
    transaction_date DATE NOT NULL DEFAULT CURRENT_DATE,
    customer_id INTEGER REFERENCES customers(id),
    employee_id INTEGER REFERENCES employee_profiles(id),
    transaction_type VARCHAR(20) CHECK (transaction_type IN ('sale', 'return', 'refund')) DEFAULT 'sale',
    payment_method VARCHAR(30) CHECK (payment_method IN ('cash', 'credit_card', 'bank_transfer', 'credit')),
    payment_status VARCHAR(20) CHECK (payment_status IN ('pending', 'paid', 'partial', 'overdue')) DEFAULT 'pending',
    subtotal DECIMAL(15,2) NOT NULL,
    tax_amount DECIMAL(15,2) DEFAULT 0,
    discount_amount DECIMAL(15,2) DEFAULT 0,
    total_amount DECIMAL(15,2) NOT NULL,
    notes TEXT,
    due_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- 7. TRANSACTION DETAILS TABLE
-- =====================================================
CREATE TABLE transaction_details (
    id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(12,2) NOT NULL,
    discount_percent DECIMAL(5,2) DEFAULT 0,
    discount_amount DECIMAL(12,2) DEFAULT 0,
    line_total DECIMAL(15,2) NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES untuk Performance
-- =====================================================
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_employee ON transactions(employee_id);
CREATE INDEX idx_transaction_details_transaction ON transaction_details(transaction_id);
CREATE INDEX idx_transaction_details_product ON transaction_details(product_id);
CREATE INDEX idx_customers_type ON customers(customer_type);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_employees_department ON employee_profiles(department);

-- =====================================================
-- VIEWS untuk Reporting
-- =====================================================

-- View untuk Sales Summary
CREATE VIEW sales_summary AS
SELECT 
    t.transaction_date,
    t.transaction_no,
    c.customer_name,
    e.first_name || ' ' || e.last_name AS employee_name,
    t.total_amount,
    t.payment_status
FROM transactions t
LEFT JOIN customers c ON t.customer_id = c.id
LEFT JOIN employee_profiles e ON t.employee_id = e.id
WHERE t.transaction_type = 'sale';

-- View untuk Product Sales
CREATE VIEW product_sales AS
SELECT 
    p.product_name,
    cat.category_name,
    SUM(td.quantity) AS total_quantity_sold,
    SUM(td.line_total) AS total_revenue
FROM transaction_details td
JOIN products p ON td.product_id = p.id
JOIN categories cat ON p.category_id = cat.id
JOIN transactions t ON td.transaction_id = t.id
WHERE t.transaction_type = 'sale'
GROUP BY p.id, p.product_name, cat.category_name;

-- =====================================================
-- TRIGGERS untuk Auto Update
-- =====================================================

-- Function untuk update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers untuk auto update timestamp
CREATE TRIGGER update_company_profiles_updated_at BEFORE UPDATE ON company_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_employee_profiles_updated_at BEFORE UPDATE ON employee_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_customers_updated_at BEFORE UPDATE ON customers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();