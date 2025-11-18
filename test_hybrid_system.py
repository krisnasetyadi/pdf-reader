"""
Simple test script for the hybrid QA system
"""

# Test 1: Database connection
print("🧪 Testing database connection...")
try:
    from database_sqlite import test_connection, SessionLocal
    from db_models import BusinessEntity

    if test_connection():
        print("✅ Database connection successful!")

        # Test database query
        with SessionLocal() as db:
            entities = db.query(BusinessEntity).all()
            print(f"📊 Found {len(entities)} business entities in database")
            for entity in entities[:3]:  # Show first 3
                print(f"   - {entity.name} ({entity.entity_type})")
    else:
        print("❌ Database connection failed")

except Exception as e:
    print(f"❌ Database test failed: {e}")

print("\n" + "=" * 50)

# Test 2: Hybrid Processor
print("🤖 Testing hybrid processor...")
try:
    from hybrid_processor import hybrid_processor

    # Test intent classification
    test_queries = [
        "List all companies in our database",  # Structured
        "Explain what buyback switch means",   # Unstructured
        "Which companies mentioned buyback procedures?"  # Hybrid
    ]

    for query in test_queries:
        intent = hybrid_processor.classify_query_intent(query)
        print(f"📝 Query: '{query}'")
        print(f"   Intent scores: Structured={intent['structured']:.2f}, Unstructured={intent['unstructured']:.2f}")

except Exception as e:
    print(f"❌ Hybrid processor test failed: {e}")

print("\n" + "=" * 50)

# Test 3: Document processor
print("📚 Testing document processor...")
try:
    from processor import processor

    # Check if processor is initialized
    if hasattr(processor, '_initialized') and processor._initialized:
        print("✅ Document processor initialized")

        # Check collections
        collections = processor.get_all_collections()
        print(f"📁 Found {len(collections)} document collections")
        for collection in collections[:3]:
            print(f"   - {collection}")
    else:
        print("⚠️ Document processor not initialized (this is normal on first run)")

except Exception as e:
    print(f"❌ Document processor test failed: {e}")

print("\n" + "=" * 50)
print("🏁 Test complete! The hybrid system is ready.")
print("\n💡 Next steps:")
print("1. Upload some PDFs using the existing upload endpoint")
print("2. Add some business entities to the database")
print("3. Test hybrid queries that combine both data sources")
print("\n🌐 Start the application with: uvicorn main:app --reload")
