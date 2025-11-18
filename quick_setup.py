"""
Quick setup script untuk database
"""


def setup_database_with_sample_data():
    print("🗄️ Setting up database...")

    from database_sqlite import SessionLocal, test_connection
    from db_models import BusinessEntity, ChatSession
    import uuid

    # Test connection
    if not test_connection():
        print("❌ Database connection failed")
        return False

    print("✅ Database connection successful")

    # Add sample data
    with SessionLocal() as db:
        # Check if data already exists
        existing_entity = db.query(BusinessEntity).first()
        if existing_entity:
            print("📊 Sample data sudah ada")
            entities = db.query(BusinessEntity).all()
            for entity in entities:
                print(f"   - {entity.name} ({entity.entity_type})")
            return True

        # Create sample entities
        sample_entities = [
            BusinessEntity(
                id=str(uuid.uuid4()),
                name="PT. Bank Central Asia",
                entity_type="bank",
                description="Bank swasta terbesar di Indonesia dengan layanan perbankan lengkap",
                attributes={"industry": "banking", "founded": "1957", "stock_code": "BBCA"}
            ),
            BusinessEntity(
                id=str(uuid.uuid4()),
                name="Buyback Switch Program",
                entity_type="financial_product",
                description="Program buyback switch adalah mekanisme dimana perusahaan dapat membeli kembali saham mereka sendiri dari pasar untuk mengelola struktur modal",
                attributes={
                    "type": "share_repurchase",
                    "purpose": "capital_management",
                    "benefits": ["increase_stock_price", "return_excess_cash", "improve_ratios"]
                }
            ),
            BusinessEntity(
                id=str(uuid.uuid4()),
                name="PT. Telkom Indonesia",
                entity_type="company",
                description="Perusahaan telekomunikasi BUMN terbesar di Indonesia",
                attributes={"industry": "telecommunications", "stock_code": "TLKM", "type": "BUMN"}
            )
        ]

        for entity in sample_entities:
            db.add(entity)

        # Create sample chat session
        session = ChatSession(
            id=str(uuid.uuid4()),
            session_name="Demo Chat",
            user_id="demo_user"
        )
        db.add(session)

        db.commit()
        print(f"✅ Added {len(sample_entities)} entities dan 1 chat session")

        # Show what was added
        for entity in sample_entities:
            print(f"   - {entity.name} ({entity.entity_type})")

    return True


if __name__ == "__main__":
    setup_database_with_sample_data()
    print("\n🎉 Database setup complete!")
    print("Anda sekarang bisa menjalankan: uvicorn main:app --reload")
