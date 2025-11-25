"""
PostgreSQL Database connection untuk business data
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

# PostgreSQL connection - akan diupdate dari setup script
DATABASE_URL = "postgresql://postgres:password@localhost:5432/business_qa_system"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL connection successful")
            return True
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False


def create_tables():
    """Create all tables (handled by SQL scripts)"""
    try:
        # Tables are created by SQL scripts, this just verifies
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully")
    except Exception as e:
        logger.error(f"Error with database tables: {e}")
        raise


def get_connection_info():
    """Get database connection information"""
    return {
        "database_type": "PostgreSQL",
        "database_url": DATABASE_URL.replace(":password@", ":***@"),  # Hide password
        "engine_info": str(engine.url)
    }
