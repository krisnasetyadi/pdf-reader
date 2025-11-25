"""
Database connection utilities for the hybrid system
"""
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)


@contextmanager
def get_business_db():
    """Context manager for business database connection"""
    db_path = Path(__file__).parent / "business_data.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Business database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


@contextmanager
def get_system_db():
    """Context manager for system database connection"""
    db_path = Path(__file__).parent / "pdf_qa.db"

    if not db_path.exists():
        raise FileNotFoundError(f"System database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def test_business_connection():
    """Test business database connection"""
    try:
        with get_business_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM company_profiles")
            result = cursor.fetchone()
            logger.info(f"Business DB: {result['count']} companies found")
            return True
    except Exception as e:
        logger.error(f"Business database connection failed: {e}")
        return False


def test_system_connection():
    """Test system database connection"""
    try:
        with get_system_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"System DB: {len(tables)} tables found: {tables}")
            return True
    except Exception as e:
        logger.error(f"System database connection failed: {e}")
        return False


def test_all_connections():
    """Test all database connections"""
    print("🧪 Testing Database Connections...")

    business_ok = test_business_connection()
    system_ok = test_system_connection()

    print(f"📊 Business Database: {'✅ Connected' if business_ok else '❌ Failed'}")
    print(f"⚙️  System Database: {'✅ Connected' if system_ok else '❌ Failed'}")

    if business_ok and system_ok:
        print("🎉 All databases connected successfully!")
        return True
    else:
        print("❌ Some database connections failed")
        return False


if __name__ == "__main__":
    test_all_connections()
