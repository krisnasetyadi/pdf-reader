"""
SQLite migration script for quick setup
"""
import logging
from database_sqlite import engine, test_connection, create_tables, reset_database
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_sqlite_database():
    """Set up SQLite database with all required tables"""
    try:
        logger.info("Setting up SQLite database...")

        # Reset database for clean start
        reset_database()

        logger.info("Database setup completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def insert_sample_data():
    """Insert some sample structured data for testing"""
    try:
        # Import after database setup
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))

        from database_sqlite import SessionLocal
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
                    name="Buyback Switch Program",
                    entity_type="financial_instrument",
                    description="Corporate finance tool allowing companies to repurchase their own shares from the market",
                    attributes={
                        "type": "share_repurchase",
                        "purpose": "capital_management",
                        "regulations": "requires_board_approval"
                    }
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
    """Main migration function for SQLite"""
    logger.info("Starting SQLite database migration...")

    # Step 1: Setup database
    if not setup_sqlite_database():
        logger.error("Database setup failed.")
        sys.exit(1)

    # Step 2: Insert sample data
    if insert_sample_data():
        logger.info("Sample data inserted successfully")

    logger.info("SQLite database migration completed successfully!")
    logger.info("Database file created: ./pdf_qa.db")
    logger.info("You can now run the application with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
