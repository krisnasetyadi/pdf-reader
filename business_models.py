"""
SQLAlchemy models untuk PostgreSQL Business Database
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean, Date, DECIMAL
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database_postgresql import Base


class CompanyProfile(Base):
    """Company profiles table"""
    __tablename__ = "company_profiles"

    id = Column(Integer, primary_key=True)
    company_name = Column(String(255), nullable=False)
    company_code = Column(String(10), unique=True, nullable=False)
    industry = Column(String(100))
    founded_year = Column(Integer)
    headquarters = Column(String(255))
    phone = Column(String(20))
    email = Column(String(100))
    website = Column(String(255))
    description = Column(Text)
    annual_revenue = Column(DECIMAL(15, 2))
    employee_count = Column(Integer)
    stock_symbol = Column(String(10))
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    employees = relationship("EmployeeProfile", back_populates="company")


class EmployeeProfile(Base):
    """Employee profiles table"""
    __tablename__ = "employee_profiles"

    id = Column(Integer, primary_key=True)
    employee_id = Column(String(20), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20))
    department = Column(String(100))
    position = Column(String(100))
    salary = Column(DECIMAL(12, 2))
    hire_date = Column(Date)
    birth_date = Column(Date)
    address = Column(Text)
    manager_id = Column(Integer, ForeignKey("employee_profiles.id"))
    company_id = Column(Integer, ForeignKey("company_profiles.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    company = relationship("CompanyProfile", back_populates="employees")
    manager = relationship("EmployeeProfile", remote_side=[id])
    transactions = relationship("Transaction", back_populates="employee")


class Customer(Base):
    """Customers table"""
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    customer_code = Column(String(20), unique=True, nullable=False)
    customer_name = Column(String(255), nullable=False)
    customer_type = Column(String(50))  # 'individual', 'corporate'
    email = Column(String(100))
    phone = Column(String(20))
    address = Column(Text)
    city = Column(String(100))
    province = Column(String(100))
    postal_code = Column(String(10))
    country = Column(String(100), default='Indonesia')
    tax_id = Column(String(30))
    credit_limit = Column(DECIMAL(15, 2))
    payment_terms = Column(Integer, default=30)  # days
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    transactions = relationship("Transaction", back_populates="customer")


class Category(Base):
    """Product categories table"""
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    category_code = Column(String(20), unique=True, nullable=False)
    category_name = Column(String(100), nullable=False)
    parent_category_id = Column(Integer, ForeignKey("categories.id"))
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    parent_category = relationship("Category", remote_side=[id])
    products = relationship("Product", back_populates="category")


class Product(Base):
    """Products table"""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    product_code = Column(String(30), unique=True, nullable=False)
    product_name = Column(String(255), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"))
    description = Column(Text)
    unit_price = Column(DECIMAL(12, 2), nullable=False)
    cost_price = Column(DECIMAL(12, 2))
    stock_quantity = Column(Integer, default=0)
    minimum_stock = Column(Integer, default=0)
    unit = Column(String(20), default='pcs')
    weight = Column(DECIMAL(8, 2))  # kg
    dimensions = Column(String(50))
    brand = Column(String(100))
    supplier = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    category = relationship("Category", back_populates="products")
    transaction_details = relationship("TransactionDetail", back_populates="product")


class Transaction(Base):
    """Sales transactions table"""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    transaction_no = Column(String(30), unique=True, nullable=False)
    transaction_date = Column(Date, nullable=False, default=func.current_date())
    customer_id = Column(Integer, ForeignKey("customers.id"))
    employee_id = Column(Integer, ForeignKey("employee_profiles.id"))
    transaction_type = Column(String(20), default='sale')  # 'sale', 'return', 'refund'
    payment_method = Column(String(30))  # 'cash', 'credit_card', 'bank_transfer', 'credit'
    payment_status = Column(String(20), default='pending')  # 'pending', 'paid', 'partial', 'overdue'
    subtotal = Column(DECIMAL(15, 2), nullable=False)
    tax_amount = Column(DECIMAL(15, 2), default=0)
    discount_amount = Column(DECIMAL(15, 2), default=0)
    total_amount = Column(DECIMAL(15, 2), nullable=False)
    notes = Column(Text)
    due_date = Column(Date)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    customer = relationship("Customer", back_populates="transactions")
    employee = relationship("EmployeeProfile", back_populates="transactions")
    transaction_details = relationship("TransactionDetail", back_populates="transaction", cascade="all, delete-orphan")


class TransactionDetail(Base):
    """Transaction details table"""
    __tablename__ = "transaction_details"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(DECIMAL(12, 2), nullable=False)
    discount_percent = Column(DECIMAL(5, 2), default=0)
    discount_amount = Column(DECIMAL(12, 2), default=0)
    line_total = Column(DECIMAL(15, 2), nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    transaction = relationship("Transaction", back_populates="transaction_details")
    product = relationship("Product", back_populates="transaction_details")
