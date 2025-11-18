# 🚀 Hybrid QA System Setup Guide

## Step 1: Install PostgreSQL

### Windows:

1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Run the installer and follow the setup wizard
3. Remember your password for the `postgres` user
4. Default port is usually 5432

### Quick Test:

```cmd
psql -U postgres -h localhost
```

## Step 2: Create Database

```sql
-- Connect as postgres user
CREATE DATABASE pdf_qa;
CREATE USER pdf_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE pdf_qa TO pdf_user;
```

## Step 3: Update Configuration

Create a `.env` file in your project root:

```env
DATABASE_URL=postgresql://pdf_user:your_password@localhost:5432/pdf_qa
```

## Step 4: Install Python Dependencies

```bash
cd d:\SYNAPSE\pdfreader
pip install -r requirements_hybrid.txt
```

## Step 5: Run Database Migration

```bash
python migrate.py
```

## Step 6: Start the Application

```bash
uvicorn main:app --reload
```

## Step 7: Test the Hybrid System

### Test Endpoints:

1. **Health Check:**

   ```bash
   curl http://localhost:8000/api/v2/health
   ```

2. **Hybrid Query:**

   ```json
   POST http://localhost:8000/api/v2/hybrid-query
   {
     "question": "Tell me about companies in our database",
     "include_sources": true
   }
   ```

3. **Create Chat Session:**
   ```bash
   curl -X POST http://localhost:8000/api/v2/chat-sessions
   ```

## Troubleshooting:

### Database Connection Issues:

- Check PostgreSQL is running: `services.msc` → PostgreSQL service
- Verify connection: `psql -U pdf_user -d pdf_qa -h localhost`
- Check firewall settings for port 5432

### Import Errors:

- Make sure all dependencies are installed
- Check Python path includes project directory

### Performance Issues:

- Ensure PostgreSQL has adequate memory allocation
- Check indexes are created properly with `\d+ table_name` in psql
