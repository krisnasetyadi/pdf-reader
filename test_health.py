#!/usr/bin/env python3
"""
Test script for the health endpoint functionality
"""
import asyncio
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processor import processor
from config import config

async def test_health_check():
    """Test the health check functionality manually"""
    print("🔍 Testing health check functionality...")
    print(f"📋 Config: Database URL configured: {bool(config.database_url)}")
    
    try:
        # Initialize processor (similar to startup)
        print("\n📦 Initializing processor...")
        processor.initialize_components()
        print("✅ Processor initialized successfully")
        
        # Test collections count
        pdf_collections = len(processor.get_all_collections()) if processor else 0
        chat_collections = len(processor.get_all_chat_collections()) if processor else 0
        
        print(f"📚 PDF collections: {pdf_collections}")
        print(f"💬 Chat collections: {chat_collections}")
        
        # Test database health
        print("\n🏥 Testing database health...")
        db_status = processor.db_manager.is_healthy() if processor and processor.db_manager else {
            "status": "not_initialized",
            "message": "Database manager not initialized", 
            "can_query": False
        }
        
        print(f"🗄️ Database status: {db_status}")
        
        # Construct health response (like the actual endpoint)
        health_response = {
            "status": "healthy" if db_status["can_query"] else "degraded",
            "initialized": hasattr(processor, '_initialized') and processor._initialized,
            "pdf_collections_count": pdf_collections,
            "chat_collections_count": chat_collections,
            "database": db_status,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n🎯 Complete health response:")
        for key, value in health_response.items():
            print(f"   {key}: {value}")
            
        return health_response
        
    except Exception as e:
        error_response = {
            "status": "unhealthy",
            "error": str(e),
            "database": {"status": "error", "message": str(e), "can_query": False},
            "timestamp": datetime.now().isoformat()
        }
        print(f"\n❌ Error during health check: {error_response}")
        return error_response

if __name__ == "__main__":
    result = asyncio.run(test_health_check())
    print(f"\n✅ Test completed! Overall status: {result['status']}")