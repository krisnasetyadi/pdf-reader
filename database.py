import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from config import Config, ALLOWED_TABLES
from typing import List, Dict, Any, Optional, Tuple
import os
import re

logger = logging.getLogger(__name__)


def validate_table_name(table_name: str) -> str:
    """Validate table name against whitelist to prevent SQL injection"""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}. Allowed: {ALLOWED_TABLES}")
    return table_name


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
                
                # Initialize Full-Text Search
                self.initialize_fts()
                
        except Exception as e:
            logger.error(f"Failed to initialize dummy data: {e}")
            self.connection.rollback()

    def initialize_fts(self):
        """Initialize Full-Text Search with tsvector columns and GIN indexes"""
        try:
            with self.connection.cursor() as cursor:
                # Create pg_trgm extension for fuzzy matching (if available)
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                    logger.info("pg_trgm extension enabled")
                except Exception as e:
                    logger.warning(f"pg_trgm extension not available: {e}")

                # Add tsvector columns and triggers for each table
                fts_configs = [
                    ('user_profiles', ['name', 'email', 'department', 'position']),
                    ('products', ['name', 'category', 'description']),
                    ('orders', ['status'])
                ]

                for table_name, text_columns in fts_configs:
                    # Check if search_vector column exists
                    cursor.execute("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = 'search_vector'
                    """, (table_name,))
                    
                    if not cursor.fetchone():
                        # Add tsvector column
                        cursor.execute(f"""
                            ALTER TABLE {table_name} 
                            ADD COLUMN IF NOT EXISTS search_vector tsvector;
                        """)
                        
                        # Build tsvector from text columns
                        coalesce_parts = " || ' ' || ".join(
                            [f"COALESCE({col}::text, '')" for col in text_columns]
                        )
                        
                        # Update existing rows
                        cursor.execute(f"""
                            UPDATE {table_name} 
                            SET search_vector = to_tsvector('indonesian', {coalesce_parts})
                            WHERE search_vector IS NULL;
                        """)
                        
                        # Create GIN index for fast search
                        cursor.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table_name}_search 
                            ON {table_name} USING GIN(search_vector);
                        """)
                        
                        logger.info(f"FTS initialized for table {table_name}")

                self.connection.commit()
                logger.info("Full-Text Search initialization completed")
                
        except Exception as e:
            logger.warning(f"FTS initialization failed (will use fallback ILIKE): {e}")
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

    def execute_query(self, query: str, params: Optional[tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
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

    def search_in_specific_tables(self, search_terms: List[str], tables: List[str], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Search across specific tables only (smart routing)"""
        logger.info(f"🔎 search_in_specific_tables called with terms: {search_terms}, tables: {tables}")
        results = {}
        
        configured_tables = Config().db_tables
        logger.info(f"🔎 Configured tables: {configured_tables}")
        
        for table_name in tables:
            if table_name not in configured_tables:
                logger.warning(f"Table {table_name} not in configured tables, skipping")
                continue
            
            logger.info(f"🔎 Searching in table: {table_name}")
            try:
                table_results = self.search_in_table(table_name, search_terms, limit)
                logger.info(f"🔎 Got {len(table_results) if table_results else 0} results from {table_name}")
                if table_results:
                    results[table_name] = table_results
                    logger.info(f"Found {len(table_results)} results in {table_name}")
            except Exception as e:
                logger.error(f"Search in table {table_name} failed: {e}")
                continue
        
        logger.info(f"🔎 Total results: {results}")
        return results
    
    # Common phrases that should be searched together
    PHRASE_PATTERNS = {
        ('project', 'manager'): 'project manager',
        ('tech', 'lead'): 'tech lead',
        ('qa', 'lead'): 'qa lead',
        ('backend', 'developer'): 'backend developer',
        ('frontend', 'developer'): 'frontend developer',
        ('devops', 'engineer'): 'devops engineer',
        ('finance', 'manager'): 'finance manager',
        ('hr', 'manager'): 'hr manager',
        ('it', 'director'): 'it director',
        ('business', 'analyst'): 'business analyst',
    }
    
    def detect_phrases(self, search_terms: List[str]) -> tuple[List[str], List[str]]:
        """Detect common phrases in search terms and return (phrases, remaining_terms)"""
        terms_lower = [t.lower() for t in search_terms]
        detected_phrases = []
        used_terms = set()
        
        for (word1, word2), phrase in self.PHRASE_PATTERNS.items():
            if word1 in terms_lower and word2 in terms_lower:
                detected_phrases.append(phrase)
                used_terms.add(word1)
                used_terms.add(word2)
        
        remaining_terms = [t for t in search_terms if t.lower() not in used_terms]
        return detected_phrases, remaining_terms
    
    # database.py - Add this method if not exists
    def search_in_table(self, table_name: str, search_terms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for terms across all columns in a table with FTS and scoring"""
        try:
            # Detect phrases first
            phrases, remaining_terms = self.detect_phrases(search_terms)
            
            # Try Full-Text Search first
            results = self.search_with_fts(table_name, search_terms, limit, phrases)
            if results:
                return results
            
            # Fallback to ILIKE if FTS fails or returns no results
            return self.search_with_ilike(table_name, search_terms, limit, phrases)
            
        except Exception as e:
            logger.error(f"Search in table {table_name} failed: {str(e)}")
            return []

    def search_with_fts(self, table_name: str, search_terms: List[str], limit: int = 10, phrases: List[str] = None) -> List[Dict[str, Any]]:
        """Full-Text Search with ts_rank scoring - prioritizes phrase matches"""
        try:
            # Validate table name against whitelist
            table_name = validate_table_name(table_name)
            
            # Check if table has search_vector column
            schema = self.get_table_schema(table_name)
            has_fts = any(col['column_name'] == 'search_vector' for col in schema)
            
            if not has_fts:
                return []
            
            # Build tsquery from search terms with OR logic
            # Each term is searched separately and combined with OR (|)
            stemmed_terms = [self.indonesian_stem(term.lower()) for term in search_terms]
            # Filter out empty terms and create tsquery format
            valid_terms = [t for t in stemmed_terms if t and len(t) >= 2]
            
            if not valid_terms:
                logger.info(f"No valid terms for FTS in {table_name}")
                return []
            
            # Use to_tsquery with OR logic: term1:* | term2:* (prefix matching)
            query_parts = [f"{term}:*" for term in valid_terms]
            tsquery_string = ' | '.join(query_parts)
            
            logger.info(f"🔍 FTS query for {table_name}: {tsquery_string}")
            
            # Get text columns for phrase matching boost
            text_columns = [col['column_name'] for col in schema 
                           if col['data_type'] in ['text', 'character varying', 'varchar']
                           and col['column_name'] != 'search_vector']
            
            # Build phrase boost expressions if phrases detected
            phrase_boost_parts = []
            phrase_params = []
            if phrases:
                for phrase in phrases:
                    for col in text_columns:
                        # Boost score by 10 if exact phrase found
                        phrase_boost_parts.append(f"CASE WHEN LOWER({col}) LIKE LOWER(%s) THEN 10.0 ELSE 0.0 END")
                        phrase_params.append(f"%{phrase}%")
            
            if phrase_boost_parts:
                phrase_boost_expr = " + ".join(phrase_boost_parts)
                query = f"""
                    SELECT *, 
                           ts_rank(search_vector, to_tsquery('simple', %s)) + ({phrase_boost_expr}) as relevance_score
                    FROM {table_name}
                    WHERE search_vector @@ to_tsquery('simple', %s)
                    ORDER BY relevance_score DESC
                    LIMIT %s
                """
                params = tuple([tsquery_string] + phrase_params + [tsquery_string, limit])
            else:
                query = f"""
                    SELECT *, 
                           ts_rank(search_vector, to_tsquery('simple', %s)) as relevance_score
                    FROM {table_name}
                    WHERE search_vector @@ to_tsquery('simple', %s)
                    ORDER BY relevance_score DESC
                    LIMIT %s
                """
                params = (tsquery_string, tsquery_string, limit)
            
            results = self.execute_query(query, params)
            
            # Post-filter: If phrases detected, prioritize exact phrase matches
            if phrases and results:
                def has_phrase_match(record):
                    for col in text_columns:
                        val = str(record.get(col, '')).lower()
                        for phrase in phrases:
                            if phrase.lower() in val:
                                return True
                    return False
                
                # Separate exact matches from partial matches
                exact_matches = [r for r in results if has_phrase_match(r)]
                partial_matches = [r for r in results if not has_phrase_match(r)]
                
                # If we have exact matches, return only those (up to limit)
                if exact_matches:
                    logger.info(f"🎯 Found {len(exact_matches)} exact phrase matches in {table_name}")
                    return exact_matches[:limit]
            
            logger.info(f"FTS found {len(results)} results in {table_name}")
            return results
            
        except Exception as e:
            logger.warning(f"FTS search failed for {table_name}: {e}")
            return []

    def search_with_ilike(self, table_name: str, search_terms: List[str], limit: int = 10, phrases: List[str] = None) -> List[Dict[str, Any]]:
        """Fallback ILIKE search with basic scoring - case insensitive, prioritizes phrase matches"""
        try:
            # Validate table name against whitelist
            table_name = validate_table_name(table_name)
            
            schema = self.get_table_schema(table_name)
            text_columns = [col['column_name'] for col in schema 
                        if col['data_type'] in ['text', 'character varying', 'varchar']
                        and col['column_name'] != 'search_vector']
            
            if not text_columns:
                return []
            
            # Build conditions for WHERE clause
            conditions = []
            where_params = []
            
            # Build score expression separately
            score_parts = []
            score_params = []
            
            # Add phrase matching with higher score (10 points per phrase match)
            if phrases:
                for column in text_columns:
                    for phrase in phrases:
                        conditions.append(f"LOWER({column}) LIKE LOWER(%s)")
                        where_params.append(f"%{phrase}%")
                        score_parts.append(f"CASE WHEN LOWER({column}) LIKE LOWER(%s) THEN 10 ELSE 0 END")
                        score_params.append(f"%{phrase}%")
            
            # Add individual term matching (1 point per term match)
            for column in text_columns:
                for term in search_terms:
                    # For WHERE clause
                    conditions.append(f"LOWER({column}) LIKE LOWER(%s)")
                    where_params.append(f"%{term}%")
                    
                    # For score calculation
                    score_parts.append(f"CASE WHEN LOWER({column}) LIKE LOWER(%s) THEN 1 ELSE 0 END")
                    score_params.append(f"%{term}%")
            
            score_expr = " + ".join(score_parts) if score_parts else "0"
            where_clause = " OR ".join(conditions)
            
            # Parameters order: score_params first, then where_params, then limit
            all_params = score_params + where_params + [limit]
            
            query = f"""
                SELECT *, ({score_expr}) as relevance_score 
                FROM {table_name} 
                WHERE {where_clause} 
                ORDER BY relevance_score DESC
                LIMIT %s
            """
            
            logger.info(f"🔍 ILIKE search in {table_name} for terms: {search_terms}, phrases: {phrases}")
            results = self.execute_query(query, tuple(all_params))
            logger.info(f"ILIKE found {len(results)} results in {table_name}")
            return results
            
        except Exception as e:
            logger.error(f"ILIKE search in table {table_name} failed: {str(e)}")
            return []

    def indonesian_stem(self, word: str) -> str:
        """Simple Indonesian stemming - remove common suffixes"""
        word = word.lower().strip()
        
        # Common Indonesian suffixes
        suffixes = ['kan', 'an', 'i', 'nya', 'lah', 'kah', 'pun']
        prefixes = ['me', 'di', 'ke', 'se', 'ber', 'ter', 'pe']
        
        # Remove suffixes
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        # Remove prefixes
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        return word  
      
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table"""
        try:
            # Validate table name against whitelist
            table_name = validate_table_name(table_name)
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

    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public';
                """)
                tables = cursor.fetchall()
                return [table['table_name'] for table in tables]
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []
db_manager = DatabaseManager()