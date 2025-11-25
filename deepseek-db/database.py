# database.py
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from config import config
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=config.db_host,
                port=config.db_port,
                database=config.db_name,
                user=config.db_user,
                password=config.db_password,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to PostgreSQL database successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

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
            logger.error(f"Failed to get schema for table {table_name}: {str(e)}")
            return []

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
            logger.error(f"Failed to get tables: {str(e)}")
            return []

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def search_in_table(self, table_name: str, search_terms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for terms across all columns in a table"""
        try:
            # Get table schema to know which columns to search
            schema = self.get_table_schema(table_name)
            text_columns = [col['column_name'] for col in schema
                            if col['data_type'] in ['text', 'character varying', 'varchar']]

            if not text_columns:
                return []

            # Build dynamic query with OR conditions across all text columns
            conditions = []
            params = []

            for column in text_columns:
                for term in search_terms:
                    conditions.append(f"{column} ILIKE %s")
                    params.append(f"%{term}%")

            where_clause = " OR ".join(conditions)
            query = f"""
                SELECT * FROM {table_name} 
                WHERE {where_clause}
                LIMIT %s
            """
            params.append(limit)

            return self.execute_query(query, tuple(params))

        except Exception as e:
            logger.error(f"Search in table {table_name} failed: {str(e)}")
            return []

    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample data from a table"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT %s"
            return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Failed to get sample from {table_name}: {str(e)}")
            return []

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Global database instance
db_manager = DatabaseManager()
