# 🗄️ PostgreSQL Business Database Setup Guide

## 📋 Apa yang Akan Dibuat

Database PostgreSQL untuk sistem hybrid QA dengan data bisnis terstruktur:

### **Tabel yang Akan Dibuat:**

1. **`company_profiles`** - Profil perusahaan (BCA, Telkom, dll)
2. **`employee_profiles`** - Data karyawan dan jabatan
3. **`customers`** - Data pelanggan (individual & corporate)
4. **`categories`** - Kategori produk (Electronics, Food, dll)
5. **`products`** - Master produk dengan stok dan harga
6. **`transactions`** - Transaksi penjualan
7. **`transaction_details`** - Detail item per transaksi

### **Sample Data yang Tersedia:**

- 5 perusahaan (BCA, Telkom, Indofood, dll)
- 5 karyawan dengan department berbeda
- 7 pelanggan (mix individual & corporate)
- 12 produk (smartphone, laptop, makanan, dll)
- 7 transaksi dengan total 47+ juta rupiah

---

## 🚀 Langkah Setup PostgreSQL

### **Step 1: Install PostgreSQL**

```bash
# Download dari: https://www.postgresql.org/download/windows/
# Atau gunakan chocolatey:
choco install postgresql
```

### **Step 2: Jalankan Setup Script**

```bash
cd d:\SYNAPSE\pdfreader
python setup_postgresql_business.py
```

Script akan meminta:

- Password PostgreSQL (user: postgres)
- Otomatis membuat database: `business_qa_system`
- Membuat semua tabel dan insert sample data

### **Step 3: Verifikasi Setup**

```bash
# Test koneksi ke database
python -c "from database_postgresql import test_connection; test_connection()"
```

### **Step 4: Update Hybrid System**

```python
# Update hybrid_processor.py untuk menggunakan PostgreSQL
from database_postgresql import SessionLocal
from business_models import CompanyProfile, Product, Transaction
```

---

## 📊 Contoh Query yang Bisa Dilakukan

### **Query Structured Data:**

- _"Perusahaan apa saja yang ada di database?"_
- _"Siapa karyawan dengan gaji tertinggi?"_
- _"Produk apa yang stoknya habis?"_
- _"Berapa total penjualan bulan November?"_
- _"Customer mana yang paling banyak berbelanja?"_

### **Query Hybrid (Structured + PDF):**

- _"Bagaimana cara kerja buyback switch di perusahaan perbankan seperti BCA?"_
- _"Jelaskan strategi penjualan produk elektronik berdasarkan data transaksi"_
- _"Apa saja regulasi yang perlu dipatuhi untuk transaksi finansial?"_

---

## 🔧 Manual Setup (Jika Script Gagal)

### **1. Buat Database Manual:**

```sql
-- Connect ke PostgreSQL sebagai superuser
psql -U postgres

-- Buat database
CREATE DATABASE business_qa_system;

-- Buat user khusus (optional)
CREATE USER qa_admin WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE business_qa_system TO qa_admin;
```

### **2. Jalankan Schema:**

```bash
psql -U postgres -d business_qa_system -f postgresql_business_schema.sql
```

### **3. Insert Sample Data:**

```bash
psql -U postgres -d business_qa_system -f postgresql_business_seed.sql
```

---

## 🔍 Testing Database

### **Cek Isi Database:**

```sql
-- Connect ke database
psql -U postgres -d business_qa_system

-- Cek jumlah record di setiap tabel
SELECT 'company_profiles' as table_name, COUNT(*) as records FROM company_profiles
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'transactions', COUNT(*) FROM transactions;

-- Lihat sample transaksi
SELECT
    t.transaction_no,
    c.customer_name,
    t.total_amount,
    t.payment_status
FROM transactions t
JOIN customers c ON t.customer_id = c.id
ORDER BY t.transaction_date DESC;
```

### **Test Hybrid Query:**

```bash
# Start aplikasi dengan PostgreSQL
uvicorn main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/api/v2/hybrid-query \
  -H "Content-Type: application/json" \
  -d '{"question": "Perusahaan apa saja yang ada?", "include_sources": true}'
```

---

## ⚙️ Configuration

### **Database Connection String:**

```
postgresql://username:password@localhost:5432/business_qa_system
```

### **Environment Variables (Optional):**

```bash
# Buat file .env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/business_qa_system
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=business_qa_system
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

---

## 🎯 Hasil Akhir

Setelah setup berhasil, Anda akan memiliki:

✅ **Database terstruktur** dengan data bisnis real
✅ **Sample data** untuk testing (47 juta+ transaksi)
✅ **Hybrid query** yang menggabungkan data terstruktur + PDF
✅ **Performance optimized** dengan index dan views
✅ **Scalable architecture** untuk produksi

### **API Endpoints Baru:**

- `POST /api/v2/hybrid-query` - Query gabungan
- `GET /api/v2/business/companies` - List perusahaan
- `GET /api/v2/business/sales` - Report penjualan
- `GET /api/v2/business/products` - Master produk

🎉 **Database siap digunakan untuk sistem QA hybrid yang powerful!**
