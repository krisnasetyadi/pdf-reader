# PostgreSQL Installation and Setup Guide

## 1. Install PostgreSQL

### Windows:
```powershell
# Option 1: Using Chocolatey
choco install postgresql

# Option 2: Download installer from postgresql.org
# Download from: https://www.postgresql.org/download/windows/
```

### Alternative - Docker (Recommended for Development):
```powershell
# Pull PostgreSQL image
docker pull postgres:15

# Run PostgreSQL container
docker run --name postgres-qa -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15

# Connect to container
docker exec -it postgres-qa psql -U postgres
```

## 2. Database Setup

### Option A: Automated Setup
```powershell
# Install required packages first
pip install psycopg2-binary

# Run automated setup
python setup_postgresql.py
```

### Option B: Manual Setup
```sql
-- 1. Create database
CREATE DATABASE pdf_qa_experiment;

-- 2. Create user
CREATE USER qa_user WITH PASSWORD 'qwerty123';
GRANT ALL PRIVILEGES ON DATABASE pdf_qa_experiment TO qa_user;

-- 3. Connect to the new database
\c pdf_qa_experiment

-- 4. Run schema setup
\i postgresql_setup.sql

-- 5. Seed data (optional)
\i business_data_seed.sql
```

## 3. Configuration

Update `config.py`:
```python
database_url = "postgresql://qa_user:qwerty123@localhost:5432/pdf_qa_experiment"
```

## 4. Verify Installation

```powershell
# Test database connection
python database_postgresql.py

# Test hybrid system
python hybrid_business_test.py
```

## 5. PostgreSQL vs SQLite Benefits

### PostgreSQL Advantages:
- **Concurrent Access**: Multiple users simultaneously
- **Advanced Search**: Full-text search with ranking
- **Performance**: Better indexing and query optimization
- **Scalability**: Handle larger datasets
- **Extensions**: UUID, trigram search, etc.
- **ACID Compliance**: Better transaction handling

### Features Enabled:
1. **UUID Primary Keys**: Better for distributed systems
2. **Full-Text Search**: `search_vector` columns with automatic updates
3. **Trigram Matching**: Fuzzy text search capabilities
4. **Performance Views**: Pre-aggregated business intelligence
5. **Advanced Indexes**: GIN indexes for array and text search

## 6. Development Workflow

```powershell
# 1. Start PostgreSQL
# If using Docker:
docker start postgres-qa

# 2. Test connection
python -c "from database_postgresql import test_connection; print('OK' if test_connection() else 'FAIL')"

# 3. Run hybrid queries
python hybrid_business_test.py

# 4. Check full-text search
python -c "from database_postgresql import test_full_text_search; test_full_text_search()"
```

## 7. Troubleshooting

### Common Issues:

1. **Connection Refused**:
   ```
   Solution: Ensure PostgreSQL is running
   Windows: Check Services.msc for PostgreSQL service
   Docker: docker start postgres-qa
   ```

2. **Authentication Failed**:
   ```
   Solution: Check password in config.py
   Default: postgres/postgres or qa_user/qwerty123
   ```

3. **Database Does Not Exist**:
   ```
   Solution: Run setup script or create manually
   python setup_postgresql.py
   ```

4. **Permission Denied**:
   ```
   Solution: Grant privileges to qa_user
   GRANT ALL PRIVILEGES ON DATABASE pdf_qa_experiment TO qa_user;
   ```

## 8. Performance Optimization

The PostgreSQL setup includes:
- GIN indexes for full-text search
- Trigger-based search vector updates
- Materialized views for business intelligence
- Connection pooling in SQLAlchemy

## 9. Backup and Restore

```powershell
# Backup
pg_dump -U qa_user -h localhost pdf_qa_experiment > backup.sql

# Restore
psql -U qa_user -h localhost pdf_qa_experiment < backup.sql
```

## 10. Next Steps

After setup:
1. Test hybrid queries combining PDF + business data
2. Experiment with full-text search ranking
3. Optimize queries using EXPLAIN ANALYZE
4. Add more business data for testing
5. Implement caching strategies