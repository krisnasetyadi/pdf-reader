"""
Database migration script to set up PostgreSQL for hybrid system
"""
import logging
from sqlalchemy import text
from db_models import Base
from database import engine, test_connection, create_tables
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    try:
        # For PostgreSQL, you might need to create the database first
        # This assumes you have the right permissions
        logger.info("Testing database connection...")
        if test_connection():
            logger.info("Database connection successful")
            return True
        else:
            logger.error("Database connection failed")
            return False
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False


def setup_database():
    """Set up the database with all required tables and indexes"""
    try:
        logger.info("Creating database tables...")
        create_tables()

        # Create additional indexes for performance
        with engine.connect() as conn:
            # Index for text search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_documents_title_search 
                ON documents USING gin(to_tsvector('english', title))
            """))

            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_business_entities_name_search 
                ON business_entities USING gin(to_tsvector('english', name))
            """))

            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_qa_pairs_question_search 
                ON qa_pairs USING gin(to_tsvector('english', question))
            """))

            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_content_search 
                ON chat_messages USING gin(to_tsvector('english', content))
            """))

            # Regular indexes for foreign keys and common queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
                ON document_chunks(document_id)
            """))

            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id 
                ON chat_messages(session_id)
            """))

            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_documents_collection_id 
                ON documents(collection_id)
            """))

            conn.commit()
            logger.info("Database indexes created successfully")

        logger.info("Database setup completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def insert_sample_data():
    """Insert some sample structured data for testing"""
    try:
        from database import SessionLocal
        from db_models import BusinessEntity, ChatSession
        import uuid

        with SessionLocal() as db:
            # Check if sample data already exists
            existing_entity = db.query(BusinessEntity).first()
            if existing_entity:
                logger.info("Sample data already exists, skipping insertion")
                return True

            # Create sample business entities
            sample_entities = [
                BusinessEntity(
                    id=str(uuid.uuid4()),
                    name="PT. Teknologi Nusantara",
                    entity_type="company",
                    description="Indonesian technology company specializing in AI and machine learning solutions",
                    attributes={"industry": "technology", "founded": "2020", "employees": "50-100"}
                ),
                BusinessEntity(
                    id=str(uuid.uuid4()),
                    name="Machine Learning Platform",
                    entity_type="product",
                    description="End-to-end ML platform for data scientists and engineers",
                    attributes={"category": "software", "pricing": "subscription", "features": ["AutoML", "Model Deployment"]}
                ),
                BusinessEntity(
                    id=str(uuid.uuid4()),
                    name="John Smith",
                    entity_type="person",
                    description="Senior Data Scientist with expertise in NLP and computer vision",
                    attributes={"role": "data_scientist", "experience": "8_years", "skills": ["Python", "TensorFlow", "PyTorch"]}
                )
            ]

            for entity in sample_entities:
                db.add(entity)

            # Create sample chat session
            sample_session = ChatSession(
                id=str(uuid.uuid4()),
                session_name="Demo Session",
                user_id="demo_user"
            )
            db.add(sample_session)

            db.commit()
            logger.info(f"Inserted {len(sample_entities)} sample entities and 1 chat session")

        return True

    except Exception as e:
        logger.error(f"Sample data insertion failed: {e}")
        return False


def main():
    """Main migration function"""
    logger.info("Starting database migration...")

    # Step 1: Test connection
    if not create_database_if_not_exists():
        logger.error("Database connection failed. Please check your DATABASE_URL configuration.")
        sys.exit(1)

    # Step 2: Create tables and indexes
    if not setup_database():
        logger.error("Database setup failed.")
        sys.exit(1)

    # Step 3: Insert sample data
    if insert_sample_data():
        logger.info("Sample data inserted successfully")

    logger.info("Database migration completed successfully!")
    logger.info("You can now run the application with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
