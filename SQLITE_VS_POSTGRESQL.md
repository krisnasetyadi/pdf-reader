# 🔄 Migration Guide: SQLite → PostgreSQL

## When to Migrate?

### Stick with SQLite if:

- ✅ Development/testing environment
- ✅ Single user access
- ✅ Data size < 100MB
- ✅ Simple queries only
- ✅ Demo purposes

### Migrate to PostgreSQL when:

- 🚀 Production deployment
- 🚀 Multiple concurrent users
- 🚀 Data size > 100MB
- 🚀 Complex queries needed
- 🚀 Network access required
- 🚀 Advanced text search needed

---

## Migration Steps

### 1. Install PostgreSQL

```bash
# Windows: Download from https://www.postgresql.org/download/
# Default: localhost:5432, user: postgres
```

### 2. Create Database

```sql
CREATE DATABASE pdf_qa_production;
CREATE USER qa_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE pdf_qa_production TO qa_user;
```

### 3. Update Configuration

```python
# config.py - Production mode
class Config:
    # ... other settings ...

    # PostgreSQL for production
    business_database_url = "postgresql://qa_user:secure_password@localhost:5432/pdf_qa_production"
    system_database_url = "postgresql://qa_user:secure_password@localhost:5432/pdf_qa_production"
```

### 4. Schema Migration

```python
# Create migration script
from sqlalchemy import create_engine
import pandas as pd

def migrate_sqlite_to_postgresql():
    # Read from SQLite
    sqlite_engine = create_engine("sqlite:///./business_data.db")

    # Write to PostgreSQL
    pg_engine = create_engine("postgresql://qa_user:password@localhost:5432/pdf_qa_production")

    tables = ['company_profiles', 'employees', 'sales_transactions', ...]

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", sqlite_engine)
        df.to_sql(table, pg_engine, if_exists='replace', index=False)
```

### 5. Update Dependencies

```bash
# Add PostgreSQL driver
pip install psycopg2-binary
```

---

## Performance Comparison

| Feature               | SQLite            | PostgreSQL        |
| --------------------- | ----------------- | ----------------- |
| **Setup Time**        | 0 minutes         | 30-60 minutes     |
| **Concurrent Users**  | 1                 | 100+              |
| **Data Size Limit**   | ~280 TB           | No limit          |
| **Query Performance** | Fast (small data) | Fast (optimized)  |
| **Full-Text Search**  | Basic FTS5        | Advanced tsvector |
| **JSON Support**      | Basic             | Native JSONB      |
| **Backup**            | Copy file         | pg_dump/restore   |
| **Monitoring**        | None              | Rich tooling      |

---

## Hybrid QA System Specific Benefits

### SQLite (Current)

```python
# Simple queries work fine
SELECT * FROM company_profiles WHERE industry = 'Technology';
SELECT COUNT(*) FROM sales_transactions;
```

### PostgreSQL (Advanced)

```sql
-- Full-text search across multiple columns
SELECT *,
       ts_rank(to_tsvector('english', company_name || ' ' || description),
               plainto_tsquery('english', 'technology software')) as rank
FROM company_profiles
WHERE to_tsvector('english', company_name || ' ' || description)
      @@ plainto_tsquery('english', 'technology software')
ORDER BY rank DESC;

-- Complex analytics
SELECT
    c.company_name,
    SUM(s.total_amount) as revenue,
    COUNT(s.id) as transactions,
    AVG(s.total_amount) as avg_transaction
FROM company_profiles c
JOIN sales_transactions s ON c.id = s.company_id
WHERE s.transaction_date >= '2023-01-01'
GROUP BY c.id, c.company_name
HAVING SUM(s.total_amount) > 100000000
ORDER BY revenue DESC;
```

---

## Current Recommendation

**Untuk sistem Anda saat ini:** **TETAP PAKAI SQLite** ✅

**Alasan:**

1. ✅ System masih development phase
2. ✅ Single user testing
3. ✅ Data size masih kecil
4. ✅ Focus on ML/AI features, bukan database optimization
5. ✅ Easy backup and portability

**Migrate ke PostgreSQL ketika:**

- 🚀 Ready untuk production deployment
- 🚀 Butuh multiple concurrent users
- 🚀 Data growth signifikan (>1000 records per table)
- 🚀 Butuh advanced text search untuk hybrid queries
- 🚀 Need network access dari aplikasi lain

**Bottom Line:** SQLite perfect untuk development, PostgreSQL untuk production! 🎯
