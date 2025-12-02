import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from config import Config
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
        self.initialize_dummy_data()

    def connect(self):
        "Establish database connection"
        try:
            self.connection = psycopg2.connect(
                host=Config().db_host,
                port=Config().db_port,
                database=Config().db_name,
                user=Config().db_user,
                password=Config().db_password,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def initialize_dummy_data(self):
        "Initialize database with dummy data for demonstration"
        try:
            with self.connection.cursor() as cursor:
                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(150) UNIQUE NOT NULL,
                        department VARCHAR(100),
                        position VARCHAR(100),
                        phone VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS products (
                        id SERIAL PRIMARY KEY, 
                        name VARCHAR(200) NOT NULL,
                        category VARCHAR(100),
                        price DECIMAL(10, 2),
                        description TEXT,
                        stock_quantity INTEGER, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY, 
                        user_id INTEGER REFERENCES user_profiles(id),
                        product_id INTEGER REFERENCES products(id),
                        quantity INTEGER,
                        total_amount DECIMAL(10, 2),
                        status VARCHAR(50),
                        order_date DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

              
                """)

                print('executed')
                cursor.execute("SELECT COUNT(*) FROM user_profiles;")

                if cursor.fetchone()['count'] == 0:
                    # Insert dummy user profiles
                    cursor.execute("""
                        INSERT INTO user_profiles (name, email, department, position, phone) VALUES
                        ('Ahmad Wijaya', 'ahmad.wijaya@company.com', 'IT', 'Software Engineer', '+62-812-3456-7890'),
                        ('Sari Dewi', 'sari.dewi@company.com', 'HR', 'HR Manager', '+62-813-4567-8901'),
                        ('Budi Santoso', 'budi.santoso@company.com', 'Finance', 'Finance Analyst', '+62-814-5678-9012'),
                        ('Maya Sari', 'maya.sari@company.com', 'Marketing', 'Marketing Specialist', '+62-815-6789-0123'),
                        ('Rizki Pratama', 'rizki.pratama@company.com', 'IT', 'System Administrator', '+62-816-7890-1234');
                    """)

                cursor.execute("SELECT COUNT(*) as count FROM products")
                if cursor.fetchone()['count'] == 0:
                    cursor.execute("""
                        INSERT INTO products (name, category, price, description, stock_quantity) VALUES
                        ('Laptop ThinkPad X1', 'Electronics', 15000000, 'Business laptop dengan processor Intel i7 dan RAM 16GB', 25),
                        ('Smartphone Galaxy S23', 'Electronics', 12000000, 'Flagship smartphone dengan kamera 108MP', 50),
                        ('Office Chair Ergonomic', 'Furniture', 2500000, 'Kursi kantor ergonomis dengan lumbar support', 15),
                        ('Project Management Software', 'Software', 5000000, 'Software manajemen proyek dengan fitur kolaborasi tim', 100),
                        ('Wireless Mouse', 'Electronics', 350000, 'Mouse nirkabel dengan precision sensor', 75);
                    """)

                cursor.execute("SELECT COUNT(*) as count FROM orders")
                if cursor.fetchone()['count'] == 0:
                    cursor.execute("""
                        INSERT INTO orders (user_id, product_id, quantity, total_amount, status, order_date) VALUES
                        (1, 1, 1, 15000000, 'completed', '2024-01-15'),
                        (2, 3, 2, 5000000, 'completed', '2024-01-16'),
                        (3, 2, 1, 12000000, 'pending', '2024-01-17'),
                        (1, 4, 1, 5000000, 'completed', '2024-01-18'),
                        (4, 5, 5, 1750000, 'shipped', '2024-01-19');
                    """)

                self.connection.commit()
                logger.info("Dummy data initialized in database")
        except Exception as e:
            logger.error(f"Failed to initialize dummy data: {e}")
            self.connection.rollback()
            

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                
                """, (table_name,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return []

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                # if cursor.description:
                return cursor.fetchall()
                # self.connection.commit()
                # return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # self.connection.rollback()
            # return []
            raise
    
    def search_accross_tables(self, search_terms: List[str], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all configured tables"""
        results = {}
       
        for table_name in Config().db_tables:
            try:
                table_results = self.search_in_table(table_name, search_terms, limit)
                if table_results:
                    results[table_name] = table_results
            except Exception as e:
                logger.error(f"Search in table {table_name} failed: {e}")
                continue
            
        return results

    # def search_in_table(self, table_name: str, search_terms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    #     """Search for accross all columns in a table"""
    #     try:
    #        schema = self.get_table_schema(table_name)
    #        text_columns = [col['column_name'] for col in schema
    #                        if col['data_type'] in ['character varying', 'text', 'varchar']]
    #        print(f"🔍 Searching in table: {table_name}")
    #        print(f"📝 Search terms: {search_terms}")
    #        print(f"📊 Text columns: {text_columns}")
    #        if not text_columns:
    #            return []

    #        conditions = []
    #        params = []

    #        for column in text_columns:
    #            for term in search_terms:
    #                conditions.append(f"{column} ILIKE %s")
    #                params.append(f"%{term}%")

    #        where_clause = " OR ".join(conditions)
    #        query = f"""
    #            SELECT * FROM {table_name}
    #            WHERE {where_clause}
    #            LIMIT %s
    #        """

    #        params.append(limit)
    #        print(f"📋 SQL Query: {query}")
    #        print(f"🔢 Query params: {params}")
    #        results = self.execute_query(query, tuple(params))
    #        print(f"✅ Found {len(results)} results in table {table_name}")
    #        return results
    #     except Exception as e:
    #         logger.error(f"Failed to search in table {table_name}: {e}")
    #         return []
    # database.py - Add this method if not exists
    def search_in_table(self, table_name: str, search_terms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for terms across all columns in a table"""
        try:
            schema = self.get_table_schema(table_name)
            text_columns = [col['column_name'] for col in schema 
                        if col['data_type'] in ['text', 'character varying', 'varchar']]
            
            if not text_columns:
                return []
            
            conditions = []
            params = []
            
            for column in text_columns:
                for term in search_terms:
                    conditions.append(f"{column} ILIKE %s")
                    params.append(f"%{term}%")
            
            where_clause = " OR ".join(conditions)
            query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT %s"
            params.append(limit)
            
            return self.execute_query(query, tuple(params))
            
        except Exception as e:
            logger.error(f"Search in table {table_name} failed: {str(e)}")
            return []  
      
    def get_table_sample(self, table_name: str,limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT %s"
            return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Failed to get sample from table {table_name}: {e}")
            return []

    def close(self):
        "Close database connection"
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

db_manager = DatabaseManager()