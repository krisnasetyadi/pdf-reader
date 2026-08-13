# Plan: Generic "Reference Framework Gap Analysis" Mode di DocuLens (ISO 27001 sebagai preset pertama)

## Context

Di meeting kemarin, ada usulan supaya DocuLens diarahkan jadi "agent yang bisa guide ISO 27001 compliance" untuk perusahaan (mulai dari kebutuhan project "Stea" yang diwajibkan ISO 27001, dan gap ke standar Amerika). User (pemilik produk) menyanggah usulan ini di meeting karena khawatir itu **menggeser fokus DocuLens** dari identitasnya sekarang: platform AI document Q&A / hybrid search umum (PDF + DB + chat log).

Pertanyaan yang mau dijawab plan ini: **bisa gak fitur ISO 27001 gap-analysis ditambahkan tanpa menggeser fokus DocuLens?** Setelah eksplorasi arsitektur backend (`pdf-reader`), jawabannya: **bisa** — dan lebih baik lagi kalau kemampuannya **digeneralisir**, bukan dibikin ISO-27001-specific. Kalau dihardcode ke ISO 27001 (nama field, prompt, struktur kontrol Annex A ke-embed di kode), itu justru yang bikin kesannya "menggeser fokus" — kode & produk jadi kelihatan dibikin buat 1 use-case doang. Kalau digeneralisir jadi kemampuan generik **"bandingkan sekumpulan dokumen ke sebuah reference framework, hasilkan temuan terstruktur per-item"**, maka ISO 27001 cuma jadi **preset/config pertama** dari kapabilitas itu — dan DocuLens tetap "document intelligence platform" yang general-purpose, malah makin general.

Klarifikasi dari user:
- Sumber standar ISO 27001: belum ada salinan resmi/berlisensi, rencana pakai **ringkasan publik** (ITGovernance, ISMS.online, dll) — perlu disclaimer eksplisit ke end-user bahwa ini interpretasi sekunder, bukan teks resmi ISO (teks resmi berbayar/berlisensi).
- Target: fitur ini masuk **di dalam DocuLens sendiri** (client-facing) — dokumen perusahaan customer di-upload & disimpan di DocuLens, lalu dicek kesesuaiannya ke framework/standar tertentu.
- Output: **keduanya** — chat Q&A biasa untuk tanya-jawab per item/kontrol, DAN laporan gap-analysis terstruktur (daftar item + status + rekomendasi).
- Diputuskan bareng: kemampuannya harus **framework-agnostic**, biar DocuLens bisa dipakai untuk kasus lain juga (SOC 2, GDPR, bandingkan SOP lama vs baru, kontrak vs template standar, dll), bukan cuma ISO 27001.
- **Positioning vs tools compliance-automation (mis. Scrut)**: sempat muncul ide supaya DocuLens integrasi API langsung ke infra (AWS/GCP/HRIS dll) kayak Scrut buat cek kontrol teknis real-time — ini disepakati **TIDAK diambil**, karena itu proyek besar terpisah (integrasi per-provider, maintenance ongoing, expertise infra/security yang beda) dan justru bikin fokus geser jadi head-to-head vs tool yang udah kuat di situ. Target pasar fitur ini secara sadar adalah **perusahaan yang belum/gak mau beli tool compliance-automation kayak Scrut** — kebutuhan mereka masih di level "dokumen kebijakan kita udah bener apa belum", bukan "config infra kita udah sesuai apa belum". Jadi pendekatan tetap **document-based** (upload kebijakan/SOP/evidence, dibandingkan ke reference framework), bukan API-integration. Kalau nanti perlu cross-check ke laporan dari tool lain (termasuk export dari Scrut), itu diperlakukan sebagai **target collection biasa** (upload file export-nya), bukan integrasi API khusus.

## Temuan arsitektur (dari eksplorasi `pdf-reader`)

- **Collection** = pasangan folder UUID (`data/uploads/{id}` + `data/indices/{id}`), content-agnostic — tidak ada field "tipe/mode" di `CollectionInfo` (models.py). Upload lewat `router/upload.py` sudah generic, jadi upload dokumen framework (ISO 27001, dll) + dokumen client bisa langsung pakai endpoint yang ada, tanpa ubah skema penyimpanan.
- **Query routing** (`processor.py` → `Processor.hybrid_search()`) sudah mendukung scoping ke `pdf_collection_ids` tertentu — cocok dipakai untuk scoping ke "collection reference" vs "collection target".
- **Prompting** di `generate_hybrid_answer()` cuma pilih dari 4 template statis (`_build_general_prompt`, `_build_aggregation_prompt`, `_build_comparison_prompt`, `_build_explanation_prompt`) berdasarkan intent pertanyaan — **belum ada** yang keyed ke collection/use-case, dan belum ada konsep "structured item-by-item comparison". Ini titik ekstensi utamanya.
- Tidak ada dokumentasi/desain existing soal "vertical mode" atau "framework comparison", jadi ini genuinely kapabilitas baru, tapi arsitekturnya (collection generic + query scoping) sudah siap nampung tanpa restrukturisasi besar.

## Rekomendasi pendekatan

Bangun kapabilitas generik **"Reference Framework Gap Analysis"** — opsional, terpisah dari alur chat default — dengan ISO 27001 sebagai **preset/config pertama**, bukan fitur yang di-hardcode.

**Konsep "Skill"**: `mode` yang tadinya cuma 1 nilai (`structured_gap_analysis`) dinaikkan jadi konsep **Skill** — tiap skill = 1 jenis perilaku analisis (prompt + output schema + reference material sendiri), dipilih user, bukan hardcoded. Ini kebutuhan nyata karena ada 2 use-case dengan perilaku beda:

| | **Skill 1: Compliance Gap Check** | **Skill 2: Scenario/Regulatory Impact** |
|---|---|---|
| Cocok buat | ISO 27001, ISO 9001 (evidence-based) | Kasus pajak/finance (contoh Moonlay) |
| Pertanyaan inti | "Requirement ini sudah dipenuhi apa belum, bukti/monitoring/test-nya mana?" | "Kalau pilih opsi A vs B, konsekuensinya apa?" |
| Input "target" | Dokumen kebijakan/SOP/evidence yang sudah ada | Deskripsi skenario/opsi transaksi (belum tentu dokumen) |
| Output | List item: status (Met/Partial/Not Met/Unknown) + evidence + rekomendasi | Perbandingan opsi: dampak pajak, dampak lain, risiko, sitasi pasal |
| Risiko/disclaimer | Standar: sumber publik, bukan teks resmi | Lebih berat: bukan nasihat pajak resmi, harus disclaim tegas |

Phase awal: bangun **plumbing skill generik** (registry: `skill_id`, nama, prompt template, output schema) + implementasikan Skill 1 (siap dipakai, ISO jadi validasi pertama) + **scaffold** Skill 2 (struktur siap, tapi konten/prompt-nya belum dianggap "siap pakai serius" sampai ada validasi risk & disclaimer terpisah — lihat bagian Keputusan).

1. **Backend (`pdf-reader`)**
   - Tambah field opsional `skill_id` di `HybridQueryRequest` (models.py) — generik, mis. `"compliance_gap_check"` / `"scenario_regulatory_impact"` (bukan nama spesifik ISO/pajak). Default tetap `general` (chat biasa), tidak breaking untuk flow existing.
   - Tambah konsep **`framework_config`** (bisa berupa file/collection metadata, bukan hardcoded di kode) yang berisi: nama framework (mis. "ISO 27001"), daftar item/kontrol yang mau dicek, kriteria status. Untuk MVP, `framework_config` ini cukup berbentuk 1 collection referensi (dokumen ringkasan kontrol) + sedikit metadata (nama framework, jumlah item) — belum perlu schema kontrol yang rigid.
   - Tambah prompt builder generik per-skill di `processor.py` (`_build_gap_check_prompt` untuk Skill 1, `_build_scenario_impact_prompt` untuk Skill 2), dipilih via skill registry berdasarkan `skill_id` — bukan if/else hardcoded per nama framework. `_build_gap_check_prompt(framework_name, reference_context, target_context)` menghasilkan output terstruktur: per item framework → status (Met/Partial/Not Met/Unknown) → kutipan bukti + sitasi sumber (pakai fitur PDF source-link yang sudah ada) → rekomendasi singkat.
   - Reuse `Processor.hybrid_search()`/`search_across_collections()` yang sudah ada — scoping query ke 2 collection: "reference framework" + "dokumen milik client/target". Endpoint generik: `router/compliance.py` → `POST /api/v1/analysis/gap-analysis` menerima `skill_id`, `reference_collection_id`, `target_collection_id(s)`, `framework_name` (parameter khusus Skill 1) — bukan endpoint yang namanya "iso27001".
   - Disclaimer wajib disematkan di response/prompt ketika reference framework-nya berasal dari sumber publik (bukan teks resmi berlisensi) — generik untuk framework apa pun, bukan cuma ISO.
   - Seed data ISO 27001 (ringkasan 93 kontrol Annex A dari sumber publik kredibel) di-upload sebagai 1 collection referensi biasa via endpoint upload existing — jadi preset pertama, tanpa kode khusus ISO.
   - **Cara mengisi reference collection**: bisa lewat upload file manual, ATAU lewat **link sumber publik yang di-attach user/admin** (reuse `upload_pdf_from_url`/`upload_pdfs_from_urls` yang sudah ada di `router/upload.py`). Sifatnya **one-time import/snapshot** — link di-fetch sekali jadi dokumen tersimpan internal, bukan live-sync otomatis yang terus re-crawl kalau sumbernya berubah. Admin yang menentukan & menempelkan link-nya secara manual, sistem tidak crawling/mencari sumber sendiri — supaya tetap konsisten dengan keputusan document-based (bukan integrasi berkelanjutan ke sumber eksternal).
   - **Persistence/tracking per item (dari phase awal, bukan phase 2)**: tiap kali gap-analysis dijalankan, simpan sebagai 1 "run" — tabel baru (via `migrations/`, dikelola lewat `database.py`) mis. `gap_analysis_runs` (id, framework_name, reference_collection_id, target_collection_id, created_at) dan `gap_analysis_items` (run_id, item_name, status, evidence, source_citation, recommendation). Schema-nya generik (gak ada kolom khusus ISO), jadi bisa dipakai reference framework apa pun. Ini yang memungkinkan compare status antar waktu ("kontrol X kemarin belum, sekarang udah") — value inti buat use-case compliance jangka panjang.

2. **Orkestrasi eksekusi skill (bukan agent otonom)** — buat Skill 1 (dan Skill 2 kalau nanti dipakai serius), jangan pakai 1 prompt tunggal untuk seluruh reference (naive RAG, gampang miss item kalau reference-nya besar mis. 93 kontrol ISO), dan jangan juga dibikin agentic loop yang LLM bebas milih langkah sendiri (mahal, lambat, gak predictable, dan completeness-nya jadi gak terjamin — padahal justru "semua item harus tercover" itu yang penting di gap-check). Pendekatannya:
   - **Map per item/batch**: loop/batch item reference (mis. 5-10 kontrol per LLM call) → retrieve evidence spesifik dari target collection per item/batch → generate verdict per item — pola deterministik di level kode, bukan diserahkan ke LLM buat mutusin urutan/langkah.
   - **Concurrency**: jalankan batch-batch ini paralel, reuse pola `ThreadPoolExecutor` yang udah ada di `Processor.hybrid_search()` (dipakai buat `_search_pdf/_search_db/_search_chat` concurrent) — sekarang diterapkan di level per-item/batch, bukan per-source, untuk kurangi latency total.
   - **Cache index reference collection** — reference (standar/framework) hampir gak pernah berubah antar run; FAISS index-nya gak perlu di-rebuild tiap run, cuma target yang perlu re-embed kalau ada dokumen baru.
   - **Incremental re-run saat iterasi** — pas user upload dokumen baru ke target (step "Iterasi" di flow), idealnya cuma re-analisis item yang relevan ke dokumen baru itu, bukan re-run semua item dari nol.
   - **Retry per item/batch, bukan retry seluruh run** — kalau parsing JSON gagal di 1 item/batch, retry item itu doang.
   - **Batas panjang evidence quote per item** di prompt, biar prompt gak bengkak kalau target collection-nya besar.
   - **Intent "meta/help" (self-describing, deterministik)**: tambah 1 intent baru di `analyze_intent()` untuk pertanyaan soal aplikasi itu sendiri (mis. "apa yang bisa dilakukan disini", "gimana cara pakai ini") — dipicu waktu user stuck gak tau langkah selanjutnya. Jawabannya **wajib digenerate dari skill registry + state user yang sebenarnya** (skill apa yang aktif, collection apa yang sudah di-upload), BUKAN dibiarkan LLM jawab bebas dari general knowledge — ini satu-satunya jenis pertanyaan di DocuLens yang gak ke-ground ke dokumen, jadi paling rawan halusinasi soal fitur yang sebenarnya gak ada kalau gak dibatasi.

3. **Frontend (`chat-ui`)**
   - Bukan halaman/produk baru — cukup **mode toggle generik** di `chat-interface.tsx` (mis. "General Q&A" vs "Gap/Compliance Analysis"), dengan dropdown pilih reference collection (yang isinya bisa ISO 27001, atau framework/dokumen lain nantinya).
   - Saat mode ini aktif, render hasil terstruktur sebagai tabel (item | status | evidence | rekomendasi) memakai komponen `ui/` yang sudah ada (table/badge) — reuse pattern dari `pdf-viewer-dialog.tsx` untuk sitasi balik ke sumber.
   - Tetap satu produk DocuLens, satu sidebar/collection list — gap-analysis mode cuma filter cara nanya + cara nampilin jawaban, generik untuk reference framework apa pun, bukan aplikasi/branding terpisah dan bukan UI khusus ISO.
   - **Onboarding/empty-state UI**: kartu saran/quick-prompt pas user baru masuk atau chat kosong (mis. "Upload dokumen & tanya jawab", "Jalankan compliance gap-check") di `chat-interface.tsx` — bantu user yang stuck gak tau mau ngapain. Ini UI statis (bukan LLM-generated), jadi selalu akurat dan gak ada resiko ngarang fitur.
   - **"/" command menu (bukan cuma sekali/one-shot help)**: quick-action list yang selalu bisa diakses (mirip slash-command Notion/Slack), jadi discovery gak nunggu user sadar buat nanya duluan. Command yang masuk akal di phase ini: `/upload` (upload ke collection), `/collections` (lihat daftar collection milik user), `/gap-check` (mulai Skill 1, guided pilih reference+target), `/history` (lihat riwayat run dari `gap_analysis_runs`), `/help` (daftar command + penjelasan singkat). `/scenario` (Skill 2) ditandai experimental atau disembunyikan dari command list publik dulu, konsisten sama status "scaffold only". Command list ini jadi cara utama discovery (deterministik, UI-driven); intent "meta/help" di chat jadi fallback buat pertanyaan bebas yang gak pakai "/".

## Alur penggunaan

### Skill 1: Compliance Gap Check (contoh pemakaian: ISO 27001 / ISO 9001)

Flow-nya sengaja gak nyebut "ISO" di step-nya — semua istilah generik (reference, target, skill/framework diisi user), supaya sama persis dipakai buat framework lain nanti. Ini urutan end-to-end-nya:

1. **Setup reference collection** — user upload dokumen "standar/framework" (mis. ringkasan ISO 27001 Annex A, atau klausul proses ISO 9001) lewat endpoint upload biasa (file manual atau link publik yang di-attach, one-time snapshot), kasih nama/label bebas (mis. "ISO 27001 Reference"). Ini collection biasa, gak ada flag khusus di sistem.
2. **Setup target collection** — user upload dokumen milik perusahaan sendiri (kebijakan, SOP, bukti-bukti proses yang udah ada) sebagai collection terpisah, mis. "Kebijakan Internal 2026".
3. **Eksplorasi bebas dulu (opsional, mode chat biasa)** — sebelum lari ke gap-analysis formal, user bisa tanya-jawab santai ("apa aja sih yang biasanya diminta framework ini soal akses kontrol?") pakai mode Q&A default yang udah ada, discope ke reference collection aja. Ini yang jawab "background issue-nya apa" dari catatan meeting kemarin — riset framework-nya lewat chat dulu, sebelum commit ke analysis formal.
4. **Jalankan gap-analysis run** — user pilih reference collection + target collection, `skill_id="compliance_gap_check"`, kasih nama framework (bebas, mis. "ISO 27001" atau "ISO 9001"), submit ke endpoint `POST /api/v1/analysis/gap-analysis`. Sistem cross-reference tiap item/klausul di reference terhadap isi target — untuk kasus proses (ISO 9001), yang dicek bukan cuma "topiknya ada dibahas", tapi eksplisit **bukti pemantauan & bukti test per klausul**. Hasilnya tabel: item | status (Met/Partial/Not Met/Unknown) | evidence & sitasi sumber | rekomendasi singkat.
5. **Simpan otomatis sebagai "run"** — hasil ini kesimpan (`gap_analysis_runs`/`gap_analysis_items`), muncul di semacam riwayat run per pasangan reference+target.
6. **Tindak lanjut per item (masih mode chat, tapi kontekstual)** — dari tabel hasil, user bisa klik/tanya lebih lanjut per item ("kenapa item ini statusnya Not Met?", "dokumen apa yang perlu ditambahin buat penuhin ini?") — reuse mode Q&A biasa, tapi discope ke item + sumber terkait. Di titik ini sistem "guide" user secara natural tanpa perlu logic khusus baru — cuma lanjutan chat dari hasil analysis.
7. **Iterasi** — user upload dokumen baru/revisi ke target collection (mis. kebijakan baru yang dibikin buat nutup gap), lalu jalankan run baru. Karena tersimpan sebagai run terpisah, bisa dibandingin manual antara run lama vs baru buat lihat progress (mana yang tadinya Not Met sekarang jadi Met).
8. **Export/report (opsional, bisa menyusul)** — run terakhir bisa di-export jadi laporan (PDF/markdown) buat dibagi ke stakeholder (mis. IT consultant, auditor) — reuse pola export/report kalau chat-ui/pdf-reader udah punya, atau ditambah belakangan, gak blocking buat MVP.

Poin pentingnya: langkah 1-8 di atas gak ada satupun yang "ISO-only" — semuanya generik by design, ISO cuma isi pertama dari nama framework dan reference collection. ISO cuma jadi driver validasi awal (karena ini kebutuhan nyata dari project Stea), bukan alasan buat bikin jalur kode terpisah.

### Skill 2: Scenario/Regulatory Impact Analysis (contoh pemakaian: pajak — kasus Moonlay)

Beda dari Skill 1: bukan "cek apakah kondisi sekarang sudah sesuai standar", tapi **"kalau saya pilih opsi A vs B, apa konsekuensinya menurut ketentuan yang berlaku"**. Target/input-nya bukan dokumen yang sudah ada untuk direview, tapi skenario/opsi yang mau dibandingkan.

1. **Setup reference collection** — user/admin upload atau attach link teks pasal/ketentuan yang relevan (mis. ketentuan PPh soal pinjaman vs modal), one-time snapshot, sama seperti Skill 1.
2. **Input skenario** — user gak upload dokumen "target" yang sudah jadi, tapi **mendeskripsikan skenario/opsi lewat chat** (mis. "investor kasih 100jt ke Moonlay, mau dibukukan sebagai pinjaman atau modal, gimana dampaknya"). Ini bisa langsung jadi query, tidak wajib ada target collection.
3. **Eksplorasi bebas dulu (mode chat biasa)** — user bisa tanya-jawab dulu soal isi pasal terkait, scoped ke reference collection, sama seperti Skill 1 step 3.
4. **Jalankan skill `scenario_regulatory_impact`** — user submit skenario + opsi-opsi yang mau dibandingkan (mis. "pinjaman" vs "modal") ke endpoint yang sama (`POST /api/v1/analysis/gap-analysis`, `skill_id` beda). Prompt-nya beda dari Skill 1: bukan schema status per item, tapi **schema per opsi** — opsi | dampak pajak | dampak lain (mis. persepsi kesehatan keuangan/bonafiditas) | risiko | sitasi pasal yang dipakai.
5. **Simpan sebagai "run"** — sama seperti Skill 1, tapi schema tabel `gap_analysis_items`-nya perlu fleksibel menampung "opsi" bukan cuma "item/kontrol" (lihat catatan skema di bagian Backend/Keputusan).
6. **Tindak lanjut per opsi (chat kontekstual)** — user bisa gali lebih lanjut ("kalau pinjaman, gimana strukturnya biar laporan keuangan tetap kelihatan bonafit?").
7. **Disclaimer wajib & tegas di setiap output** — ini BUKAN nasihat pajak/hukum resmi, hasil analisis AI berbasis dokumen yang diupload, keputusan akhir harus dikonsultasikan ke konsultan pajak/akuntan bersertifikat. Ini lebih berat dari disclaimer Skill 1 karena dampak salahnya nyata (denda pajak, dsb) — lihat juga poin risiko di bagian Keputusan.
8. **Belum masuk fase "siap pakai serius"** — beda dari Skill 1 (yang divalidasi lewat kasus ISO 27001/project Stea), Skill 2 di fase awal ini statusnya **scaffold/exploratory**: struktur skill & endpoint disiapkan, tapi belum direkomendasikan dipakai untuk keputusan pajak nyata sampai ada validasi terpisah soal akurasi, sumber pasal yang dipakai, dan review dari pihak yang punya kompetensi pajak/hukum.

## File kritis

- `pdf-reader/models.py` — `HybridQueryRequest`, `CollectionInfo`
- `pdf-reader/processor.py` — `generate_hybrid_answer`, 4 prompt builder existing, tempat tambah prompt builder generik per-skill (mis. `_build_gap_check_prompt`, `_build_scenario_impact_prompt`), dipilih lewat skill registry, bukan if/else hardcoded
- `pdf-reader/storage.py` — `register_collection` (kalau mau nambah label "reference framework" vs "target" di collection)
- `pdf-reader/router/upload.py`, `router/collections.py` — dipakai apa adanya untuk upload/attach-link dokumen reference & target
- `pdf-reader/router/compliance.py` (baru) — thin wrapper endpoint generik untuk gap-analysis, terima `skill_id`
- `pdf-reader/migrations/`, `pdf-reader/database.py` — tabel baru `gap_analysis_runs` & `gap_analysis_items` untuk persistence/tracking (skema `items` perlu fleksibel: kolom umum seperti `label`/`status`/`evidence`/`recommendation` dipakai bersama oleh kedua skill, bukan kolom khusus "kontrol" vs "opsi")
- `chat-ui/components/chat-interface.tsx` — tempat nambah skill toggle/dropdown + pemilihan reference collection
- `chat-ui/services/` (`endpoint.ts`, `types.ts`) — tambah field `skill_id`, `reference_collection_id` di request type & endpoint call

## Keputusan

- **Sumber ISO 27001**: belum ada, akan dicari nanti (kandidat: ringkasan publik ITGovernance/ISMS.online/dll) — cek lisensi/hak pakai sebelum dipakai jadi seed collection.
- **Rilis**: internal dulu. Karena semua masih di environment development/staging, aman untuk push cepat dan iterasi tanpa risiko ke production/client lain dulu.
- **Struktur `framework_config` di fase awal**: mulai dari 1 collection referensi + nama framework (belum perlu skema kontrol yang rigid/predefined di luar itu). Response dari `_build_structured_comparison_prompt` wajib dalam **bentuk terstruktur** (list item: nama kontrol, status, evidence, rekomendasi — bukan paragraf bebas).
- **Tracking/history per item: dimasukkan dari phase awal** (bukan ditunda ke fase 2) — karena masih tahap development/staging, gak ada ruginya langsung simpan tiap hasil gap-analysis sebagai "run" yang persisten (lihat detail tabel di bagian Backend di atas), supaya history/progress antar waktu bisa langsung ditrack sejak awal.
- **Verifikasi kualitas structured output** — sebelum dianggap "siap dipakai", perlu diverifikasi dulu: (1) konsistensi schema antar run untuk input yang sama, (2) status enum gak melenceng dari set tetap (Met/Partial/Not Met/Unknown), (3) akurasi evidence/citation balik ke sumber dokumen, (4) schema tetap generik saat dicoba dengan reference framework berbeda (bukan cuma ISO 27001), (5) reliability parsing JSON dari output Gemini — tetap perlu retry/repair logic per item kalau parsing gagal, walau cuma 1 provider.
- **Provider LLM: Gemini-only** — HuggingFace dan Ollama sudah tidak dipakai (HF terlalu lambat, Ollama belum/gak akan disetup). Jadi `get_llm()` untuk skill ini cukup diarahkan ke Gemini saja, gak perlu logic fallback antar-provider. Ini juga bikin structured/JSON output lebih reliable dari awal (Gemini lebih konsisten soal ini dibanding model open-source lokal).
- **Konsep Skill diadopsi**: `mode` digeneralisir jadi `skill_id` (registry sederhana, bukan enum tetap di kode) supaya nampung 2 perilaku analisis berbeda — Skill 1 (Compliance Gap Check) dan Skill 2 (Scenario/Regulatory Impact) — tanpa duplikasi endpoint/infra. Detail flow masing-masing skill ada di bagian "Alur penggunaan".
- **Skill 2 (pajak/finance) hanya di-scaffold, belum untuk dipakai serius** — karena sifatnya advisory (bukan sekadar cek bukti seperti Skill 1) dan salahnya berdampak nyata (denda pajak dsb.), Skill 2 dibangun strukturnya (skill registry, endpoint, prompt generik) di phase ini, tapi konten/prompt spesifik pajak-nya butuh validasi terpisah (akurasi, sumber pasal, review dari pihak berkompeten pajak/hukum) sebelum dianggap layak dipakai untuk keputusan nyata — beda status dengan Skill 1 yang sudah divalidasi lewat kebutuhan project Stea.

## Tinjauan: trade-off, nilai jual, dan cek pergeseran fokus

**Konsistensi vs pertanyaan awal ("apakah menggeser fokus DocuLens")**: Tidak bergeser. Identitas produk tetap document-intelligence platform umum; kapabilitas baru ini opsional (skill), reuse ~80% infra yang sudah ada (collection, hybrid search, PDF citation), dan sengaja menolak 2 kali kesempatan untuk melebar (integrasi infra ala Scrut, dan crawling eksternal otomatis). ISO 27001 & pajak cuma jadi contoh pemakaian ("skill instance"), bukan identitas baru.

**Tapi scope-nya membesar dari ide awal ("tambah 1 mode ISO")** — worth disadari sebagai trade-off, bukan berarti salah arah:
- Dari "tambah 1 mode" → jadi: konsep Skill generik + 2 skill + tabel persistence baru + orchestrator (batching/concurrency/caching/incremental re-run) + endpoint baru. Ini investasi rekayasa yang riil, bukan quick toggle.
- **Rekomendasi**: untuk rilis pertama (internal, masih staging), pertimbangkan **trim scope**: bangun Skill 1 penuh (termasuk orkestrasi dasar: batching + concurrency), tapi untuk Skill 2 cukup scaffold struktur skill/endpoint TANPA orkestrasi canggih (caching, incremental re-run) dulu — karena Skill 2 belum boleh dipakai serius, optimasi di situ belum berguna sampai konten & akurasinya divalidasi. Ini mencegah over-investasi di sesuatu yang belum tentu dipakai.

**Extension masa depan yang sudah "gratis" karena desain sekarang** (bukan dikerjakan sekarang, cuma dicatat supaya sadar potensinya):
- **Analytics/dashboard dari histori skill** — karena `gap_analysis_runs`/`gap_analysis_items` sudah disimpan terstruktur dari phase awal, begitu ada cukup data usage, dashboard (tren status membaik dari waktu ke waktu, item yang paling sering "Not Met" lintas run, dll) bisa dibangun cuma dengan query aggregate di atas tabel yang sudah ada — gak perlu re-arsitektur. Baru worth dikerjakan setelah ada cukup histori run nyata untuk dianalisis, bukan sekarang.
- **Intent "meta/help"** juga membuka jalan buat onboarding yang makin pintar nanti (mis. saran skill yang relevan berdasarkan collection yang sudah di-upload user) — tapi versi awalnya cukup deterministik dulu, sesuai poin orkestrasi di atas.

**Nilai jual / daya saing:**
- Dibanding chat generik (ChatGPT/chat biasa): unggul di grounding+citation, histori/tracking antar waktu, skala multi-dokumen, kontrol akses per collection — sudah dibahas di diskusi sebelumnya, tetap valid.
- Dibanding Scrut/tool compliance-automation: **bukan pengganti**, tapi pelengkap untuk segmen yang belum butuh/mampu infra-monitoring — value proposition-nya jelas dan defensible selama positioning ini dikomunikasikan dengan benar (jangan dijual sebagai "setara Scrut, lebih murah").
- **Risiko ke daya jual**: use-case compliance itu high-stakes — kalau kualitas gap-analysis-nya buruk (item ke-skip, status salah, evidence gak akurat) sekali aja ketauan, trust user ke fitur ini (dan mungkin ke DocuLens secara umum) turun drastis. Ini alasan kuat kenapa langkah verifikasi kualitas (poin 8-9 di bawah, dan Keputusan soal validasi) tidak boleh dilewat sebelum future rilis ke luar internal.
- Belum ada validasi pasar/kompetitor eksplisit (siapa lagi yang main di segmen "compliance dokumen buat perusahaan yang belum siap Scrut") — kalau mau lebih yakin soal daya jual, worth dicek terpisah, di luar scope teknis plan ini.

## Verifikasi (setelah implementasi nanti)

1. Upload collection referensi ISO 27001 (ringkasan publik) + 1 collection dummy "kebijakan perusahaan" via endpoint upload existing.
2. Panggil endpoint gap-analysis dengan `skill_id="compliance_gap_check"`, `framework_name="ISO 27001"` + kedua collection id, cek output berupa daftar item dengan status & sitasi ke dokumen yang bener (bandingkan manual beberapa kontrol).
3. Jalankan query yang sama 2-3x pada input identik, cek konsistensi schema & status enum antar run (gak ada field/kategori status yang berubah-ubah acak) — ini validasi reliability Gemini buat structured output.
4. Cek record tersimpan di `gap_analysis_runs`/`gap_analysis_items` sesuai hasil query, dan bisa di-query lagi buat lihat history run sebelumnya.
5. Coba sekali lagi dengan reference collection & framework_name yang beda (mis. dokumen SOP internal sebagai "reference") untuk memastikan prompt, endpoint, dan schema tabel memang generik per `skill_id`, bukan diam-diam ISO-specific.
6. Test chat mode default (`skill_id` tidak diisi) tetap berjalan sama seperti sebelumnya (no regression di 4 prompt template lama).
7. Di chat-ui, test toggle skill gap-analysis menampilkan tabel dengan benar untuk minimal 2 reference framework berbeda, dan mode default tetap chat biasa.
8. Cek orkestrasi: jalankan gap-analysis dengan reference collection besar (puluhan item), pastikan semua item tercover (gak ada yang ke-skip karena retrieval top-k biasa), dan ukur latency dengan/tanpa concurrency buat konfirmasi manfaat batching+paralel.
9. Skill 2 (`scenario_regulatory_impact`) diuji cuma sebatas smoke-test struktur (endpoint & schema jalan), TIDAK divalidasi buat akurasi konten pajak — sesuai keputusan "scaffold only".
