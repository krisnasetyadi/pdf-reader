# Diskusi: Redesign Source Panel ala NotebookLM + Non-Admin Self-Upload

Status: **draft diskusi, belum untuk dieksekusi**

## Konteks

Saat ini halaman `/sources` (`chat-ui/components/sources-panel.tsx`) adalah satu halaman besar dengan 4 tab terpisah: PDF File, Public Link, Chat (.txt), Database. Upload lewat tombol → file picker klasik (tidak ada drag-and-drop). Setiap source (PDF collection, chat collection, public link, koneksi DB) punya status `active`/`inactive` yang menentukan apakah dipakai saat query, ditoggle lewat badge di panel dan chip pendek di UI chat (`source-chip.tsx`).

User ingin UX yang lebih mirip NotebookLM/Gemini: source management yang lebih sederhana dan approachable, dan idenya adalah non-admin user boleh menambah dokumen sendiri, tapi **tidak untuk Database dan Chat** (karena Chat rencananya akan diisi otomatis lewat integrasi Telegram nanti, dan Database kemungkinan tetap sensitif/admin-only).

## Temuan penting dari kode saat ini (yang mengubah bentuk rencana)

Ini bukan sekadar "ganti UI" — ada 3 gap struktural yang harus ditutup dulu:

1. **Auth ada tapi tidak dipakai di mana pun yang relevan.** Backend (`pdf-reader/router/auth.py`) sudah punya JWT + kolom `role` (`user`/`admin`), dan helper `get_current_user` / `require_role("admin")` sudah diekspor — tapi **tidak ada satu pun route upload/collections/public_links/database_connections/chat yang memakainya**. Hari ini siapa pun yang bisa mengakses API bisa upload, hapus, atau toggle collection apa pun. `chat-ui/middleware.ts` juga secara eksplisit mematikan proteksi route ("Allow all paths... for testing/development").
2. **Tidak ada konsep kepemilikan (ownership).** `CollectionInfo`, `ChatCollection`, `UploadResponse` di `models.py` tidak punya field `user_id`/`owner_id`. `PublicLinkSource` dan `DatabaseConnectionSource` punya `workspace_id` opsional tapi tidak dipakai untuk filtering di mana pun. Artinya semua collection itu global/shared, bukan per-user.
3. **Telegram belum ada sama sekali** — cuma ada nilai enum `ChatPlatform.TELEGRAM` untuk label file `.txt` yang diupload manual. Belum ada bot, webhook, token config, atau dependency terkait.

Jadi "non-admin bisa upload dokumen sendiri, tapi tidak untuk DB/Chat" itu bukan cuma soal sembunyikan 2 tab di UI — itu butuh: (a) auth benar-benar ditegakkan di route upload & collections, (b) kolom kepemilikan ditambahkan supaya bisa membedakan "dokumen saya" vs "dokumen orang lain", dan (c) role-check di backend (bukan cuma UI) supaya non-admin memang tidak bisa hit endpoint Database/Chat meski lewat API langsung.

## Yang perlu didiskusikan/diputuskan sebelum ditulis jadi plan implementasi

### 1. Model kepemilikan & visibilitas
Waktu non-admin upload PDF sendiri, siapa yang bisa lihat/query dokumen itu?
- **Opsi A — Privat per user**: dokumen non-admin hanya terlihat & terpakai oleh dia sendiri (dan admin bisa lihat semua). Paling mirip NotebookLM (notebook personal).
- **Opsi B — Shared workspace**: semua dokumen (dari admin maupun non-admin) masuk pool bersama yang bisa dipakai siapa saja di workspace, hanya kontrol *tambah/hapus* yang dibatasi per role.
- **Opsi C — Campuran**: non-admin punya "My Sources" privat + bisa lihat "Shared Sources" (curated admin) sebagai referensi, tidak bisa edit yang shared.

### 2. Model interaksi source (toggle aktif vs checkbox per-query)
NotebookLM pakai checkbox per-sumber yang dicentang untuk *sesi/percakapan saat itu*, bukan status global aktif/nonaktif yang mengubah state semua orang. Sistem sekarang pakai `active`/`inactive` yang levelnya "collection", global untuk semua query berikutnya (dan berpotensi memengaruhi user lain kalau shared). Apakah kita:
- tetap pakai model `active`/`inactive` per collection (lebih sederhana, minim perubahan backend), atau
- pindah ke model "selected sources per percakapan" (lebih mirip NotebookLM, tapi perlu state per-session bukan per-collection, dan `SourceChip`/`chat-interface.tsx` perlu dirombak)?

### 3. Batasan non-admin
- Ada limit jumlah dokumen / ukuran total per non-admin user? (sekarang ada limit per-upload: 20 file/section, 3MB/file — tapi tidak ada limit akun keseluruhan)
- Non-admin boleh hapus dokumennya sendiri? Boleh lihat dokumen non-admin lain?
- Public Link (paste URL) apakah statusnya sama dengan PDF (boleh untuk non-admin) atau masuk kategori "sensitif" seperti Database?

### 4. Auth enforcement scope
Karena `middleware.ts` mematikan proteksi route dan backend belum menegakkan JWT di route manapun kecuali reset-password admin — mengaktifkan fitur ini otomatis berarti kita **menyalakan login wajib** untuk pertama kalinya secara nyata. Ini perubahan besar di luar sekadar "source panel". Perlu disepakati: apakah ini bagian dari scope sekarang, atau fitur source panel dulu jalan tanpa auth (role dummy/asumsi) sambil auth enforcement jadi task terpisah?

### 5. Telegram (untuk konteks jangka panjang, tidak perlu diputuskan sekarang)
Kalau nanti bot Telegram mengisi Chat collections otomatis, itu kemungkinan besar jalan sebagai service/admin-level integration (bot punya token per admin/workspace), bukan per-end-user upload. Ini mendukung keputusan "Chat tetap admin-only" — user biasa tidak perlu (dan sebaiknya tidak bisa) upload file chat manual kalau nanti sumber kebenarannya adalah bot yang sinkron otomatis.

## Arah desain yang saya sarankan (untuk didiskusikan, bukan final)

Mengadaptasi pola NotebookLM tanpa merombak semuanya sekaligus:

- **Satu panel source per sesi/notebook** (bukan halaman terpisah besar) dengan list source flat (bukan tab terpisah), tiap item punya icon jenis (PDF/Link), nama, dan checkbox "include in this chat" — ini yang paling mengubah rasa "app besar" jadi "notebook ringan".
- **Tombol tambah source tunggal** ("+ Add source") yang membuka pilihan: Upload PDF atau Paste Link — **Database dan Chat sengaja tidak muncul di sini** untuk non-admin (secara UI maupun backend, keduanya cek role).
- Admin tetap punya akses penuh ke semua jenis source lewat panel yang sama (atau panel admin terpisah untuk Database/Chat agar tidak membingungkan non-admin).
- Kepemilikan: rekomendasi mulai dari **Opsi A (privat per user)** dulu — paling simpel untuk diimplementasikan & paling aman secara default, bisa dilonggarkan ke shared workspace belakangan kalau dibutuhkan.
- Auth enforcement dinyalakan sebagai bagian dari fitur ini (tidak masuk akal punya "punya dokumen sendiri" tanpa identitas user yang nyata).

## Keputusan yang sudah diambil (hasil diskusi)

1. **Kepemilikan: Privat per user.** Dokumen yang diupload non-admin hanya terlihat & terpakai oleh dia sendiri. Admin bisa lihat semua. → butuh kolom `user_id`/`owner_id` di PDF collection & public link, dan filtering query berdasarkan user yang login.
2. **Model interaksi source: Checkbox per sesi percakapan** (bukan `active`/`inactive` global per collection). Ini pola NotebookLM asli — source dipilih per-chat/notebook, bukan status yang berlaku untuk semua query berikutnya. Konsekuensi:
   - Perlu state baru: daftar source yang dicentang untuk sesi/percakapan yang sedang berjalan (kemungkinan disimpan di level sesi chat, mirip `SessionsApi` yang sudah ada untuk histori).
   - `source-chip.tsx` dan bagian relevan `chat-interface.tsx` perlu dirombak — ini bukan cuma ganti label toggle, tapi ganti model data (dari "status collection" ke "seleksi per sesi").
   - Backend `hybrid_search`/`agnostic_query` sudah menerima `collection_ids` eksplisit per-request (lihat `HybridQueryRequest`), jadi secara backend ini sebenarnya **lebih cocok** dengan model checkbox-per-sesi dibanding model `active`/`inactive` yang ada sekarang — perubahan utama ada di frontend + cara collection di-resolve (privat milik user, bukan semua collection global).
3. **Auth: masuk scope sekarang**, ditambah **guest route yang dibatasi**. Artinya ada 2 kelas akses:
   - **User login (non-admin/admin)**: sesuai poin 1 & 2 di atas.
   - **Guest (belum login)**: tetap bisa pakai aplikasi tapi dibatasi. Perlu didetailkan (lihat pertanyaan terbuka di bawah) — kemungkinan besar: guest bisa chat/query tapi hanya terhadap source publik/curated (bukan privat siapa pun), tidak bisa upload dokumen sendiri, dan kena rate limit (misal per IP atau per session token sementara) untuk mencegah abuse ke Gemini API yang berbayar.

## Keputusan tambahan (putaran diskusi kedua)

4. **Guest: upload & chat sementara, tanpa persist.** Guest (belum login) tetap bisa upload dokumen dan tanya-jawab, tapi semuanya scoped ke sesi browser saat itu — begitu sesi berakhir (tab ditutup / expired), dokumen & histori dihapus, tidak masuk penyimpanan permanen (Supabase Storage/DB). Implikasi teknis:
   - Perlu identitas sesi guest sementara (mis. signed cookie/token acak yang dibuat saat pertama kali akses tanpa login — bukan JWT user asli, tapi cukup untuk scoping index & rate limit).
   - Collection guest disimpan di lokasi terpisah/temporer (mis. folder index dengan TTL, atau in-memory) dan **wajib** ada cleanup job (TTL-based), supaya tidak menumpuk jadi kebocoran disk di server.
   - **Rate limit tetap wajib** meski ephemeral — upload+embed+LLM call semuanya membebani resource yang sama (CPU/RAM index, kuota Gemini API), jadi guest tanpa batas jumlah upload/query per sesi akan jadi vektor abuse yang nyata. Perlu batas eksplisit (mis. maks N dokumen, ukuran total, dan N query per jam per sesi guest).
   - Karena ini fitur baru yang cukup besar (bukan cuma "hide tombol"), kemungkinan besar akan dikerjakan sebagai tahap terpisah setelah source panel utama (privat-per-user, login) selesai & stabil — bukan sekaligus di rilis pertama. Ini akan saya usulkan sebagai fase terpisah di plan implementasi.
5. **Migrasi data lama**: semua PDF/chat collection & public link yang sudah ada (belum punya `user_id`) di-assign menjadi milik admin pertama yang terdaftar di sistem.
6. **Public Link**: ikut aturan privat-per-user sama seperti PDF — non-admin boleh menambah Public Link miliknya sendiri. Hanya **Database** yang tetap admin-only.

## Ringkasan model akses final

| Source type | Admin | Non-admin (login) | Guest |
|---|---|---|---|
| PDF upload | semua + kelola apa saja | privat miliknya sendiri | upload sementara, tanpa persist, dibatasi rate limit |
| Public Link | semua + kelola apa saja | privat miliknya sendiri | kemungkinan diblokir (perlu dikonfirmasi saat detail fase guest) |
| Database | admin-only | tidak bisa akses | tidak bisa akses |
| Chat (.txt) | admin-only (nanti diisi otomatis via bot Telegram) | tidak bisa akses | tidak bisa akses |

## Putaran diskusi ketiga — model sharing ala Google Drive

Keputusan berubah dari "privat murni per user" jadi **model sharing eksplisit**, mirip Google Drive:

1. **Ownership + sharing, bukan cuma privat/publik biner.** Setiap dokumen punya satu **owner** (siapa yang upload). Owner (terutama admin) bisa **assign/share dokumen ke satu atau banyak user tertentu** (multi-select, seperti "Share with..." di Drive). Non-admin yang di-share bisa memakai dokumen itu untuk query, tapi tidak memilikinya.
   - Implikasi data model: butuh tabel relasi terpisah, bukan cuma kolom `user_id` tunggal — mis. `document_shares(document_id, shared_with_user_id)` (many-to-many), di atas `owner_id` di tabel collection.
   - Non-admin tetap bisa upload dokumen sendiri (privat, sesuai keputusan sebelumnya) — model sharing ini berlaku di atas itu, bukan menggantikannya. Jadi tiap dokumen non-admin **melihat**: (a) miliknya sendiri, (b) yang di-share ke dia oleh siapa pun (biasanya admin).
   - Admin tetap bisa lihat/kelola semua dokumen (privilese admin), terlepas dari status share.

2. **Model seleksi source: checkbox per sesi** — dikonfirmasi ulang, tetap seperti keputusan sebelumnya.

3. **Hak hapus**: non-admin **tidak bisa hapus** dokumen yang di-share ke dia oleh admin — hanya bisa **request delete**. Untuk dokumen miliknya sendiri (yang dia upload sendiri), tetap boleh hapus (asumsi default, silakan koreksi kalau beda).
   - Rekomendasi saya untuk "request delete": buat ini **ringan** dulu — field `deletion_requested_at`/`deletion_requested_by` di record dokumen, lalu admin melihat badge "N permintaan hapus" di panel kelola sumber. Tidak perlu sistem notifikasi/messaging penuh di iterasi pertama; itu overkill untuk kebutuhan ini dan bisa ditambah belakangan kalau volumenya besar.

4. **Public Link untuk non-admin — rekomendasi saya: izinkan, dengan syarat.** Risikonya beda dari upload file: server melakukan *outbound request* ke URL yang diberikan user, jadi ada risiko SSRF (user bisa masukkan URL yang mengarah ke jaringan internal/metadata server, bukan cuma dokumen publik). Kode `_download_remote_pdf` saat ini (`pdf-reader/router/upload.py`) tidak melakukan validasi terhadap IP privat/internal sebelum fetch. Jadi rekomendasi: **izinkan untuk non-admin**, tapi jadikan **validasi SSRF (blokir IP privat/loopback/link-local, whitelist skema http/https saja)** sebagai prasyarat keamanan sebelum fitur ini dibuka ke non-admin — bukan opsional. Ini sejalan dengan direction awal (mempermudah user tambah sumber sendiri) selama pagar keamanannya ada.

## Putaran diskusi keempat — dua perluasan scope baru

Dua hal yang disebutkan terakhir ini masing-masing cukup besar untuk jadi **fase/inisiatif terpisah**, bukan bagian dari Fase 1 (source panel + auth + sharing):

### A. Upload PDF → generalisasi jadi "File Upload" multi-format
Ingin mendukung upload CSV, PDF, TXT, XLSX, Word/DOCX, dll — bukan cuma PDF. Ini technically nontrivial karena:
- Loader/parser berbeda per format (PDF pakai `PyPDFLoader` sekarang; butuh loader terpisah untuk DOCX — mis. `python-docx`/`Docx2txtLoader`, XLSX/CSV — `pandas`/`UnstructuredExcelLoader`, TXT sudah ada `TextLoader`).
- Strategi chunking mungkin perlu beda per tipe (dokumen tabular seperti CSV/XLSX lebih masuk akal diringkas per baris/tabel daripada di-chunk seperti teks bebas — mirip cara `query_structured_data` menangani data database terstruktur, bukan seperti PDF).
- Validasi & deteksi tipe file, error handling per format, dependency tambahan di `requirements.txt`.
- Ini **cukup besar untuk jadi fase sendiri** (Fase 3), tidak menghalangi Fase 1 mulai duluan dengan PDF + Public Link saja.

### B. Integrasi Telegram — bot beneran, bukan upload file chat
Klarifikasi penting: yang diinginkan adalah **integrasi langsung/live ke Telegram** (bot yang connect ke suatu grup/channel dan mengalirkan pesan masuk secara otomatis ke Chat collection), **bukan** flow upload manual file `.txt` hasil export chat yang ada sekarang. Ini secara efektif proyek integrasi baru:
- Perlu bot Telegram (token per admin/workspace), webhook endpoint publik yang menerima update dari Telegram, dan mapping "bot ini terhubung ke chat collection mana".
- Perlu keputusan: sync historis (ambil histori lama sekali di awal) vs sync berkelanjutan (setiap pesan baru masuk otomatis), penyimpanan token bot dengan aman, serta izin (siapa yang boleh connect/disconnect bot — sejalan dengan keputusan "Chat tetap admin-only").
- Ini **jelas fase terpisah** (Fase 4), perlu sesi diskusi/plan sendiri karena berbeda karakter dari fitur source panel (bukan CRUD dokumen, tapi integrasi realtime dengan sistem eksternal).

## Ringkasan model akses (diperbarui)

| Source type | Admin | Non-admin (login) | Guest |
|---|---|---|---|
| File upload (PDF, dst) | upload, kelola semua, bisa share ke user tertentu | upload privat sendiri + akses yang di-share admin ke dia (tidak bisa hapus yang di-share, hanya request delete) | upload sementara, tanpa persist, rate-limited |
| Public Link | sama seperti file upload | privat sendiri + syarat validasi SSRF | kemungkinan diblokir (detail nanti) |
| Database | admin-only | ❌ | ❌ |
| Chat (integrasi Telegram nanti) | admin-only, connect/disconnect bot | ❌ | ❌ |

## Struktur fase yang saya usulkan untuk plan implementasi (diperbarui)

- **Fase 1 (inti)**: aktifkan auth beneran (login wajib, `middleware.ts`, JWT dicek di backend) + model ownership+sharing (`owner_id` + tabel `document_shares` many-to-many) + migrasi data lama ke admin pertama + role guard backend untuk Database & Chat (admin-only) + source panel baru dengan checkbox per-sesi + Public Link untuk non-admin (dengan validasi SSRF sebagai prasyarat) + flow "request delete" ringan untuk dokumen shared.
- **Fase 2 (guest)**: sesi guest sementara, index/storage temporer + TTL cleanup, rate limiting, UI khusus guest ("coba tanpa daftar").
- **Fase 3 (generalisasi file upload)**: dukungan CSV/XLSX/DOCX/TXT dst di luar PDF, loader & strategi chunking per tipe.
- **Fase 4 (integrasi Telegram)**: bot real-time, webhook, mapping ke Chat collection — didiskusikan & di-plan terpisah karena berbeda karakter (integrasi eksternal, bukan CRUD dokumen).

## Putaran diskusi kelima — kelola akses & target share

1. **Perlu "Manage access" penuh dari awal** (bukan share sekali-jalan). Artinya Fase 1 butuh layar/dialog tersendiri untuk melihat & mengubah siapa saja yang punya akses ke suatu dokumen kapan pun — mirip Google Drive: buka dokumen → "Manage access" → tambah/hapus user atau grup, lihat daftar yang sudah punya akses.
2. **Target share: individual user + grup/role.** Admin bisa share ke user spesifik satu-satu, **atau** ke grup/role sekaligus (mis. "semua non-admin"). Sistem role saat ini cuma punya 2 nilai (`user`/`admin` — lihat `pdf-reader/router/auth.py`), jadi untuk kebutuhan ini kemungkinan besar bentuk grupnya adalah **role-based** ("share ke semua user dengan role non-admin") di iterasi pertama, bukan grup custom yang bisa dibuat bebas (mis. "Tim Marketing") — kecuali memang itu yang dimaksud. Perlu konfirmasi singkat: grup yang dimaksud itu **role yang sudah ada** (admin/non-admin), atau grup **custom yang bisa dibuat sendiri** (butuh entitas baru "groups" + keanggotaan, lebih besar scope-nya)?

## Putaran diskusi keenam — grup custom

Dikonfirmasi: **grup custom yang bisa dibuat sendiri** (bukan sekadar role admin/non-admin). Ini menambah entitas data baru yang belum ada sama sekali di sistem sekarang:
- Tabel `groups` (id, name, created_by, dst) + `group_members` (group_id, user_id) — many-to-many.
- UI untuk admin: buat/hapus grup, tambah/hapus anggota.
- Model sharing jadi 3 arah: dokumen bisa di-share ke **user individual**, **grup**, atau kombinasi keduanya — resolusi akses saat query jadi: `is_owner OR user_id in document_shares OR user_id in (anggota grup yang ada di document_group_shares)`.

Karena ini nambah cukup banyak kerja (entitas baru + UI manajemennya), saya bagi Fase 1 jadi 3 sub-tahap yang bisa dikerjakan & diverifikasi berurutan, supaya tidak jadi satu perubahan raksasa sekaligus:

- **Fase 1a — Auth & fondasi**: nyalakan login wajib (`middleware.ts` + JWT dicek di backend semua route relevan), migrasi collection lama ke admin pertama, role guard Database/Chat (admin-only).
- **Fase 1b — Groups & Sharing**: entitas `groups`/`group_members`, tabel `document_shares` (user & grup), UI "Manage access" (mirip Drive: tambah/hapus user atau grup, lihat siapa saja yang punya akses), flow "request delete" ringan.
- **Fase 1c — Source panel baru**: UI checkbox-per-sesi, unified "+ Add source" (PDF + Public Link untuk non-admin dengan validasi SSRF), integrasi dengan hasil 1a & 1b (dokumen yang muncul = milik sendiri + yang di-share ke user/grupnya).

Fase 2 (guest), Fase 3 (multi-format upload), Fase 4 (Telegram) tetap seperti sebelumnya, dikerjakan setelah 1a–1c selesai.

## Sisa pertanyaan terbuka (detail kecil, tidak menghalangi mulai Fase 1a)

- Default checkbox source untuk sesi baru (semua tercentang vs kosong).
- Angka pasti rate limit guest (masuk Fase 2, tidak perlu diputuskan sekarang).

Kerangka fase sudah cukup matang untuk mulai ditulis jadi plan implementasi konkret. Langkah berikutnya: saya buatkan plan implementasi detail untuk **Fase 1a (Auth & fondasi)** lewat plan mode — ini fondasi yang wajib duluan sebelum 1b/1c bisa jalan (groups/sharing/source panel semua butuh identitas user yang nyata) — supaya file/fungsi yang diubah bisa direview sebelum eksekusi.
