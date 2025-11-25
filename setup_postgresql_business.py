"""
PostgreSQL Business Database Setup Script
Untuk mengatur database bisnis dengan data transaksi, profil, dll.
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import sys
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'user': 'postgres',  # Default PostgreSQL user
    'password': 'your_password_here',  # CHANGE THIS!
    'database': 'business_qa_system'
}


def create_database():
    """Create the business database if it doesn't exist"""
    print("🗄️ Creating database...")

    try:
        # Connect to PostgreSQL server (without specific database)
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['database'],))
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            print(f"✅ Database '{DB_CONFIG['database']}' created successfully")
        else:
            print(f"📊 Database '{DB_CONFIG['database']}' already exists")

        cursor.close()
        conn.close()
        return True

    except psycopg2.Error as e:
        print(f"❌ Error creating database: {e}")
        return False


def run_sql_file(filepath, description):
    """Execute SQL file"""
    print(f"🔧 {description}...")

    try:
        # Connect to the business database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Read and execute SQL file
        with open(filepath, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        cursor.execute(sql_content)
        conn.commit()

        cursor.close()
        conn.close()
        print(f"✅ {description} completed successfully")
        return True

    except psycopg2.Error as e:
        print(f"❌ Error in {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ SQL file not found: {filepath}")
        return False


def test_database_connection():
    """Test database connection and show sample data"""
    print("🧪 Testing database connection...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Test query - count records in each table
        test_queries = [
            ("company_profiles", "SELECT COUNT(*) FROM company_profiles"),
            ("employee_profiles", "SELECT COUNT(*) FROM employee_profiles"),
            ("customers", "SELECT COUNT(*) FROM customers"),
            ("products", "SELECT COUNT(*) FROM products"),
            ("transactions", "SELECT COUNT(*) FROM transactions"),
            ("transaction_details", "SELECT COUNT(*) FROM transaction_details")
        ]

        print("📊 Database contents:")
        for table_name, query in test_queries:
            cursor.execute(query)
            count = cursor.fetchone()[0]
            print(f"   - {table_name}: {count} records")

        # Show sample sales data
        print("\n💰 Sample sales summary:")
        cursor.execute("""
            SELECT 
                t.transaction_no,
                c.customer_name,
                t.total_amount,
                t.payment_status
            FROM transactions t
            LEFT JOIN customers c ON t.customer_id = c.id
            ORDER BY t.transaction_date DESC
            LIMIT 5
        """)

        sales = cursor.fetchall()
        for sale in sales:
            print(f"   - {sale[0]}: {sale[1]} - Rp {sale[2]:,} ({sale[3]})")

        cursor.close()
        conn.close()
        print("✅ Database connection test successful")
        return True

    except psycopg2.Error as e:
        print(f"❌ Database connection test failed: {e}")
        return False


def update_config_file():
    """Update config.py with PostgreSQL connection string"""
    db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

    config_content = f'''"""
Updated config.py untuk PostgreSQL Business Database
"""
import os


class Config:
    # Existing configuration
    model_name = "google/flan-t5-large"
    embedding_model = "all-MiniLM-L6-v2"
    
    # Document processing
    chunk_size = 1000
    chunk_overlap = 200
    
    # Paths
    upload_folder = "data/uploads"
    index_folder = "data/indices"
    
    # Query settings
    k_results = 5
    k_per_collection = 3
    relevance_threshold = 0.3
    
    # PostgreSQL Business Database Configuration
    database_url = "{db_url}"
    
    # Hybrid system settings
    structured_weight = 0.6
    unstructured_weight = 0.4
    max_hybrid_results = 10
    
    # PostgreSQL connection settings
    db_pool_size = 5
    db_max_overflow = 10
    db_pool_recycle = 3600


config = Config()
'''

    with open('config_postgresql.py', 'w', encoding='utf-8') as f:
        f.write(config_content)

    print("✅ Configuration file 'config_postgresql.py' created")


def main():
    """Main setup function"""
    print("🚀 PostgreSQL Business Database Setup")
    print("=" * 50)

    # Check if SQL files exist
    schema_file = Path("postgresql_business_schema.sql")
    seed_file = Path("postgresql_business_seed.sql")

    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        sys.exit(1)

    if not seed_file.exists():
        print(f"❌ Seed file not found: {seed_file}")
        sys.exit(1)

    # Get password from user
    if DB_CONFIG['password'] == 'your_password_here':
        password = input("Enter PostgreSQL password for user 'postgres': ")
        DB_CONFIG['password'] = password

    print(f"📍 Connecting to PostgreSQL server at {DB_CONFIG['host']}:{DB_CONFIG['port']}")

    # Step 1: Create database
    if not create_database():
        print("❌ Failed to create database. Exiting.")
        sys.exit(1)

    # Step 2: Create schema (tables, indexes, etc.)
    if not run_sql_file(schema_file, "Creating database schema"):
        print("❌ Failed to create schema. Exiting.")
        sys.exit(1)

    # Step 3: Insert seed data
    if not run_sql_file(seed_file, "Inserting seed data"):
        print("❌ Failed to insert seed data. Exiting.")
        sys.exit(1)

    # Step 4: Test database
    if not test_database_connection():
        print("❌ Database test failed. Please check your setup.")
        sys.exit(1)

    # Step 5: Update configuration
    update_config_file()

    print("\n🎉 PostgreSQL Business Database Setup Complete!")
    print("=" * 50)
    print("📋 What's been created:")
    print("   ✅ Database: business_qa_system")
    print("   ✅ Tables: 7 business tables with relationships")
    print("   ✅ Sample data: Companies, employees, customers, products, transactions")
    print("   ✅ Views: sales_summary, product_sales")
    print("   ✅ Indexes: Performance optimized")
    print("   ✅ Triggers: Auto-update timestamps")
    print("   ✅ Config file: config_postgresql.py")
    print("\n🔗 Connection string:")
    print(f"   postgresql://{DB_CONFIG['user']}:***@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print("\n🚀 Next steps:")
    print("   1. Update your hybrid_processor.py to use PostgreSQL")
    print("   2. Install: pip install psycopg2-binary")
    print("   3. Run: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
