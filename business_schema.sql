-- =============================================
-- DATABASE SCHEMA UNTUK SISTEM HYBRID QA
-- Tabel-tabel operasional perusahaan
-- =============================================

-- 1. TABEL PROFIL PERUSAHAAN
CREATE TABLE IF NOT EXISTS company_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    company_name VARCHAR(255) NOT NULL,
    company_code VARCHAR(20) UNIQUE NOT NULL,
    industry VARCHAR(100) NOT NULL,
    founded_year INTEGER,
    headquarters VARCHAR(255),
    total_employees INTEGER,
    annual_revenue DECIMAL(15,2),
    stock_symbol VARCHAR(10),
    is_public BOOLEAN DEFAULT 0,
    website VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. TABEL DEPARTEMEN
CREATE TABLE IF NOT EXISTS departments (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    department_name VARCHAR(255) NOT NULL,
    department_code VARCHAR(20) UNIQUE NOT NULL,
    company_id VARCHAR(36) NOT NULL,
    manager_name VARCHAR(255),
    budget DECIMAL(12,2),
    employee_count INTEGER DEFAULT 0,
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES company_profiles(id)
);

-- 3. TABEL KARYAWAN/PROFIL
CREATE TABLE IF NOT EXISTS employee_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    employee_code VARCHAR(20) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    position VARCHAR(100),
    department_id VARCHAR(36),
    company_id VARCHAR(36) NOT NULL,
    hire_date DATE,
    salary DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'active', -- active, inactive, terminated
    manager_id VARCHAR(36),
    birth_date DATE,
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (department_id) REFERENCES departments(id),
    FOREIGN KEY (company_id) REFERENCES company_profiles(id),
    FOREIGN KEY (manager_id) REFERENCES employee_profiles(id)
);

-- 4. TABEL PRODUK/LAYANAN
CREATE TABLE IF NOT EXISTS products (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    product_code VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    stock_quantity INTEGER DEFAULT 0,
    minimum_stock INTEGER DEFAULT 0,
    supplier_name VARCHAR(255),
    description TEXT,
    status VARCHAR(20) DEFAULT 'active', -- active, discontinued
    company_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES company_profiles(id)
);

-- 5. TABEL CUSTOMER/KLIEN
CREATE TABLE IF NOT EXISTS customers (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    customer_code VARCHAR(20) UNIQUE NOT NULL,
    customer_name VARCHAR(255) NOT NULL,
    customer_type VARCHAR(50), -- individual, corporate
    email VARCHAR(255),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    credit_limit DECIMAL(12,2) DEFAULT 0,
    current_balance DECIMAL(12,2) DEFAULT 0,
    registration_date DATE,
    status VARCHAR(20) DEFAULT 'active', -- active, inactive, blacklisted
    company_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES company_profiles(id)
);

-- 6. TABEL TRANSAKSI PENJUALAN
CREATE TABLE IF NOT EXISTS sales_transactions (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    transaction_no VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(36) NOT NULL,
    employee_id VARCHAR(36), -- sales person
    transaction_date DATE NOT NULL,
    subtotal DECIMAL(12,2) NOT NULL,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    discount_amount DECIMAL(10,2) DEFAULT 0,
    total_amount DECIMAL(12,2) NOT NULL,
    payment_method VARCHAR(50), -- cash, credit, transfer, etc
    payment_status VARCHAR(20) DEFAULT 'pending', -- pending, paid, overdue
    notes TEXT,
    company_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (employee_id) REFERENCES employee_profiles(id),
    FOREIGN KEY (company_id) REFERENCES company_profiles(id)
);

-- 7. TABEL DETAIL TRANSAKSI PENJUALAN
CREATE TABLE IF NOT EXISTS sales_transaction_details (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    transaction_id VARCHAR(36) NOT NULL,
    product_id VARCHAR(36) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    discount_percent DECIMAL(5,2) DEFAULT 0,
    line_total DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES sales_transactions(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- 8. TABEL INVENTORI/STOCK
CREATE TABLE IF NOT EXISTS inventory_movements (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    product_id VARCHAR(36) NOT NULL,
    movement_type VARCHAR(20) NOT NULL, -- in, out, adjustment
    quantity INTEGER NOT NULL,
    reference_no VARCHAR(100), -- PO number, sales number, etc
    movement_date DATE NOT NULL,
    reason VARCHAR(255),
    employee_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (employee_id) REFERENCES employee_profiles(id)
);

-- 9. TABEL KONFIGURASI SISTEM
CREATE TABLE IF NOT EXISTS system_config (
    id VARCHAR(36) PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    description TEXT,
    category VARCHAR(50),
    updated_by VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- INDEXES UNTUK PERFORMA
-- =============================================

CREATE INDEX IF NOT EXISTS idx_employees_company ON employee_profiles(company_id);
CREATE INDEX IF NOT EXISTS idx_employees_department ON employee_profiles(department_id);
CREATE INDEX IF NOT EXISTS idx_sales_customer ON sales_transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_sales_employee ON sales_transactions(employee_id);
CREATE INDEX IF NOT EXISTS idx_products_company ON products(company_id);
CREATE INDEX IF NOT EXISTS idx_customers_company ON customers(company_id);
CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory_movements(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory_movements(movement_date);

-- Text search indexes (untuk SQLite FTS jika tersedia)
-- CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(product_name, description, content=products, content_rowid=rowid);
-- CREATE VIRTUAL TABLE IF NOT EXISTS customers_fts USING fts5(customer_name, content=customers, content_rowid=rowid);