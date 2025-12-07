import psycopg2
from psycopg2.extras import RealDictCursor

def test_database_directly():
    try:
        connection = psycopg2.connect(
            host="localhost",
            port=5432,
            database="qa_system",
            user="postgres",
            password="qwerty123",
            cursor_factory=RealDictCursor
        )
        
        with connection.cursor() as cursor:
            # Test 1: Check if table exists and has data
            cursor.execute("SELECT COUNT(*) as count FROM user_profiles")
            count = cursor.fetchone()['count']
            print(f"📊 Total users in user_profiles: {count}")
            
            # Test 2: Check specific user
            cursor.execute("SELECT * FROM user_profiles WHERE name ILIKE '%ahmad%'")
            ahmad_users = cursor.fetchall()
            print(f"👤 Users with 'ahmad' in name: {len(ahmad_users)}")
            for user in ahmad_users:
                print(f"  - {user}")
            
            # Test 3: Check all users
            cursor.execute("SELECT * FROM user_profiles LIMIT 5")
            all_users = cursor.fetchall()
            print("👥 All users sample:")
            for user in all_users:
                print(f"  - {user}")
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")

if __name__ == "__main__":
    test_database_directly()