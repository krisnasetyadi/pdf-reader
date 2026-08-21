from config import config, LLMProvider, AVAILABLE_MODELS
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
import logging
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import os
import re
import json
import tempfile
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from models import SearchType, DatabaseResult, SourceInfo
from database import db_manager

logger = logging.getLogger(__name__)

# Off-topic scope guard: shown verbatim when a question is unrelated to both
# the retrieved context and the platform itself (e.g. vacation recommendations,
# coding help, trivia). Kept out of the failure_patterns list in
# _validate_and_clean_answer() on purpose — must not overlap those substrings
# or this canned reply gets silently replaced by a forced document answer.
OFF_TOPIC_REDIRECT_ID = (
    "Saya dirancang untuk menjawab pertanyaan seputar dokumen, database, dan "
    "chat log Anda — bukan topik umum. Coba tanyakan misalnya "
    "\"ringkas dokumen saya\" atau \"apa isi chat log saya?\""
)
OFF_TOPIC_INSTRUCTION_ID = (
    f"- Jika PERTANYAAN adalah permintaan pengetahuan umum yang tidak berkaitan "
    f"dengan dokumen di atas maupun dengan platform ini (misalnya rekomendasi "
    f"liburan, bantuan coding, trivia, atau saran pribadi), JANGAN mencoba "
    f"menjawabnya. Balas persis dengan: \"{OFF_TOPIC_REDIRECT_ID}\""
)


class PDFQAProcessor:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store_cache = {}
        self.bm25_cache = {}
        self._cache_lock = threading.RLock()
        self._initialized = False
        self._init_lock = threading.Lock()
        self.db_manager = None
        self._db_initialized = False
        
        # Multi-LLM support
        self._llm_cache = {}  # Cache for loaded LLMs: {provider_model: llm_instance}
        self._current_provider = None
        self._current_model = None

        self.query_expansion_terms = {
            "apa itu": ["definisi", "pengertian", "arti", "makna", "jelaskan"],
            "proses": ["tahapan", "langkah", "mekanisme", "cara kerja"],
            "auction": ["lelang", "penawaran", "bidding", "tender"],
            # LPDU document keywords
            "buyback": ["pembelian kembali", "buyback cash", "buyback debt switch"],
            "lpdu": ["layanan perdagangan dealer utama", "dealer utama"],
            "lpksbn": ["lelang pembelian kembali surat berharga negara"],
            "settlement": ["penyelesaian transaksi", "setelmen"],
            "staple bonds": ["paket staple", "destination series", "source series"],
            "quotation": ["kuotasi", "penawaran", "quote"],
            "allocation": ["alokasi", "pemenang lelang"]
        }

        # Add database-related query expansion
        self.db_query_expansion_terms = {
            "user": ["pengguna", "karyawan", "staff", "employee", "profil"],
            "product": ["produk", "barang", "item", "inventory"],
            "order": ["pesanan", "pembelian", "transaksi", "orderan", "pemesanan"],
            "price": ["harga", "cost", "biaya", "tarif", "nilai"],
            "quantity": ["jumlah", "kuantitas", "banyak", "stock"]
        }
        
        # Document-specific keywords (for PDF search priority)
        self.pdf_keywords = [
            'lpdu', 'lpksbn', 'buyback', 'debt switch', 'auction', 'lelang',
            'sun', 'sbn', 'djppr', 'mofids', 'dealer utama', 'settlement',
            'quotation', 'kuotasi', 'alokasi', 'staple bonds', 'securities',
            'fungsional', 'persyaratan', 'kode a', 'lpdu-bcds', 'lpdu-sa', 'lpdu-dssb',
            'maker checker', 'enrich data', 'plte', 'bank indonesia',
            'sop', 'prosedur', 'standar', 'approval', 'deployment', 'testing',
            'development', 'sdlc', 'requirement', 'design', 'maintenance'
        ]

        # Smart table routing - map keywords to specific tables
        # NOTE: Use generic keywords, NOT hardcoded values like names
        self.table_keywords = {
            "user_profiles": [
                'user', 'pengguna', 'karyawan', 'staff', 'employee', 'profil',
                'nama', 'email', 'department', 'position', 'jabatan', 'pegawai',
                'siapa', 'orang', 'anggota', 'member', 'kontak', 'telepon', 'phone',
                'divisi', 'departemen', 'bagian'  # department-related
            ],
            "products": [
                'product', 'produk', 'barang', 'item', 'harga', 'price',
                'stock', 'stok', 'kategori', 'category', 'jual', 'beli',
                'license', 'software', 'tools', 'service'
            ],
            "orders": [
                'order', 'pesanan', 'pembelian', 'transaksi', 'beli', 'pesan',
                'status', 'pending', 'completed', 'shipped', 'quantity',
                'total', 'amount', 'tanggal', 'invoice', 'processing', 'proyek'
            ]
        }
        
        # Person name patterns (untuk deteksi nama orang tanpa hardcode)
        self.person_question_patterns = [
            'siapa', 'who is', 'nama', 'karyawan bernama', 'user bernama',
            'cari orang', 'find person', 'profile', 'kontak', 'contact',
            'email', 'telepon', 'phone'
        ]
        
        # Tambahkan patterns untuk intent detection yang lebih baik
        self.intent_patterns = {
            'data_retrieval': [
                r'berapa\s+(harga|jumlah|total|stock)',
                r'(harga|price)\s+[a-zA-Z0-9\s]+',
                r'siapa\s+yang',
                r'(cari|tampilkan|lihat)\s+(data|informasi)',
                r'^[a-zA-Z]+\s+\d+$',  # Pattern seperti "Laptop 5"
            ],
            'comparison': [
                r'(bandingkan|perbandingan|vs\.?)',
                r'(lebih|paling)\s+(murah|mahal|banyak|sedikit)',
                r'mana\s+yang\s+(lebih|paling)',
            ],
            'aggregation': [
                r'(total|jumlah|rata-rata|rerata|average|sum)\s+[a-zA-Z]',
                r'berapa\s+total',
                r'hitung\s+(total|jumlah)',
                r'(semua|semua\s+data)\s+[a-zA-Z]',
            ],
            'explanation': [
                r'apa\s+itu',
                r'jelaskan\s+tentang',
                r'definisi\s+dari',
                r'bagaimana\s+cara',
                r'proses\s+[a-zA-Z]',
            ]
        }

        # "Meta/help" — questions about the APP ITSELF (not document content),
        # e.g. "apa yang bisa dilakukan disini". Checked separately from
        # intent_patterns above because it must short-circuit the whole
        # document-grounded pipeline (see is_meta_help_query/build_meta_help_answer):
        # this is the one question type with no document to ground an LLM
        # answer in, so it's kept deterministic instead of free-generated.
        self.meta_help_patterns = [
            r'apa\s+(saja\s+)?yang\s+bisa\s+(kamu|kau|anda|aplikasi\s+ini|doculens)',
            r'apa\s+yang\s+bisa\s+(saya|kita)\s+lakukan\s+di\s*sini',
            r'bisa\s+ngapain\s*(di\s*sini)?\s*\??$',
            r'cara\s+(pakai|menggunakan|make)\s+(aplikasi|ini|doculens)',
            r'fitur\s+apa\s+(saja\s+)?(yang\s+)?(ada|tersedia)',
            r'command\s+apa\s+(saja\s+)?(yang\s+)?(ada|tersedia)',
            r'^/?help$',
            r'apa\s+yang\s+bisa\s+dilakukan\s+di\s*sini',
        ]

        # Known "/" commands (must mirror chat-ui's SLASH_COMMANDS). Backend
        # guard only — the frontend intercepts unmatched slash input before
        # it ever reaches this API, this exists for direct API calls/bypass.
        self.known_slash_commands = {"/gap-check", "/collections", "/history", "/upload", "/help"}

        # Enhanced table routing
        self.enhanced_table_keywords = {
            "user_profiles": {
                "keywords": ['user', 'pengguna', 'karyawan', 'staff', 'employee', 'profil'],
                "fields": ['name', 'email', 'department', 'position', 'phone'],
                "query_types": ['siapa', 'cari orang', 'kontak', 'pegawai']
            },
            "products": {
                "keywords": ['product', 'produk', 'barang', 'item', 'harga', 'price'],
                "fields": ['name', 'category', 'price', 'description', 'stock_quantity'],
                "query_types": ['harga', 'stok', 'barang', 'produk']
            },
            "orders": {
                "keywords": ['order', 'pesanan', 'pembelian', 'transaksi'],
                "fields": ['quantity', 'total_amount', 'status', 'order_date'],
                "query_types": ['transaksi', 'pesanan', 'pembelian', 'order']
            }
        }
    
    def analyze_intent(self, question: str) -> Dict[str, Any]:
        """Enhanced intent analysis"""
        question_lower = question.lower()
        
        detected_intents = []
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    detected_intents.append(intent_type)
                    break
        
        # Determine primary intent
        primary_intent = 'unknown'
        if detected_intents:
            # Priority: aggregation > comparison > data_retrieval > explanation
            priority_order = ['aggregation', 'comparison', 'data_retrieval', 'explanation']
            for intent in priority_order:
                if intent in detected_intents:
                    primary_intent = intent
                    break
        
        return {
            "primary_intent": primary_intent,
            "all_intents": detected_intents,
            "is_aggregation": 'aggregation' in detected_intents,
            "is_comparison": 'comparison' in detected_intents,
            "is_data_retrieval": 'data_retrieval' in detected_intents,
            "is_explanation": 'explanation' in detected_intents
        }

    def is_meta_help_query(self, question: str) -> bool:
        """True if the question is about the app itself (e.g. "apa yang bisa
        dilakukan disini"), not about uploaded document content. Callers
        should short-circuit the normal hybrid-search pipeline entirely and
        use build_meta_help_answer() instead — see meta_help_patterns above
        for why this must stay deterministic."""
        q = question.lower().strip()
        return any(re.search(p, q, re.IGNORECASE) for p in self.meta_help_patterns)

    def build_meta_help_answer(self, pdf_collection_count: int, pdf_collection_titles: List[str]) -> str:
        """Deterministic capability summary, generated from the skill
        registry + the caller's real state (their own collections) — never
        from LLM general knowledge, so it can't claim a feature that doesn't
        exist."""
        if pdf_collection_count > 0:
            names = ", ".join(pdf_collection_titles[:5])
            if pdf_collection_count > 5:
                names += ", …"
            collections_line = f"Kamu punya **{pdf_collection_count} collection dokumen** aktif: {names}."
        else:
            collections_line = (
                "Kamu belum punya collection dokumen — upload dulu lewat panel "
                "**Sources** di sidebar (atau ketik `/upload`)."
            )

        return (
            "**Yang bisa dilakukan di DocuLens:**\n\n"
            "- Tanya-jawab bebas atas dokumen PDF, database, dan chat log yang sudah kamu upload\n"
            "- **Compliance Gap Check** (`/gap-check`) — bandingkan dokumen perusahaan ke standar/framework "
            "apa pun (contoh: ISO 27001, ISO 9001), hasilnya status per item + rekomendasi, bisa didownload "
            "jadi laporan PDF/markdown\n"
            "- Lihat daftar collection kamu (`/collections`) atau riwayat gap-analysis (`/history`)\n\n"
            f"{collections_line}\n\n"
            "_Skill lain (analisis skenario/regulasi, mis. kasus pajak) masih tahap pengembangan — "
            "belum tersedia untuk dipakai serius._\n\n"
            "Ketik `/` di kolom chat kapan saja untuk lihat semua command."
        )

    def is_unknown_slash_command(self, question: str) -> bool:
        """True kalau pesan diawali '/' tapi bukan salah satu command yang
        dikenal. Jaring pengaman untuk request yang bypass frontend's
        command-menu (mis. panggilan API langsung) — jalur normal lewat UI
        sudah ditangkap di chat-ui sebelum sampai ke sini."""
        q = question.strip()
        if not q.startswith('/'):
            return False
        first_token = q.split()[0].lower()
        return first_token not in self.known_slash_commands

    def build_unknown_command_answer(self, question: str) -> str:
        """Deterministic reply for an unrecognized '/' command — same spirit
        as build_meta_help_answer: never LLM-generated, so it can't
        hallucinate a command that doesn't exist."""
        attempted = question.strip().split()[0]
        commands = "\n".join(f"- `{c}`" for c in sorted(self.known_slash_commands))
        return f"Command `{attempted}` tidak dikenali.\n\n**Command yang tersedia:**\n\n{commands}\n\nKetik `/help` untuk detail."

    def expand_query(self, query):
        """Expand query with synonyms and related terms"""
        expanded_queries = [query]

        # Add lowercase version
        expanded_queries.append(query.lower())

        # Add synonyms for common terms
        for term, synonyms in self.query_expansion_terms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(term, synonym)
                    expanded_queries.append(expanded_query)
                    # Also try with original case
                    expanded_queries.append(query.replace(term, synonym))

        return list(set(expanded_queries))

    def get_vector_store(self, collection_id):
        """Get vector store from cache, local disk, or Supabase Storage."""
        with self._cache_lock:
            if collection_id in self.vector_store_cache:
                logger.debug(f"Returning cached vector store for {collection_id}")
                return self.vector_store_cache[collection_id]

            logger.info(f"🔍 Loading vector store for collection: {collection_id}")
            index_path = os.path.join(config.index_folder, collection_id)

            # ── If index not on disk, try Supabase download ────────────────
            if not os.path.exists(os.path.join(index_path, "index.faiss")):
                try:
                    import storage as supabase_storage
                    if supabase_storage.is_enabled():
                        logger.info(f"📥 Downloading index from Supabase: {collection_id}")
                        ok = supabase_storage.download_index(collection_id, index_path)
                        if not ok:
                            logger.warning(f"❌ Supabase download incomplete for {collection_id}")
                except Exception as dl_err:
                    logger.warning(f"Supabase download error: {dl_err}")

            if not os.path.exists(index_path):
                logger.warning(f"❌ Index path not found: {index_path}")
                return None

            if not all(os.path.exists(os.path.join(index_path, f))
                       for f in ["index.faiss", "index.pkl"]):
                logger.error(f"❌ Incomplete index files for {collection_id}")
                return None

            if self.embeddings is None:
                logger.error("❌ Embeddings model not loaded!")
                return None

            try:
                logger.info(f"🔄 Loading FAISS vector store from {index_path}")
                vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Successfully loaded vector store for {collection_id}")

                try:
                    test_results = vector_store.similarity_search("test", k=1)
                    logger.info(f"🧪 Test search returned {len(test_results)} results")
                except Exception as test_e:
                    logger.error(f"❌ Test search failed: {test_e}")
                    return None

                self.vector_store_cache[collection_id] = vector_store
                return vector_store

            except Exception as e:
                logger.error(
                    f"❌ Failed to load vector store for {collection_id}: {str(e)}"
                )
                import traceback
                logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                return None

    def search_across_collections(self, query, collection_ids=None, top_k=5):
        """Enhanced search with query expansion and better scoring"""
        if collection_ids is None:
            collection_ids = self.get_all_collections()

        if not collection_ids:
            logger.warning("❌ No collections available for search")
            return []

        logger.info(f"🔍 Starting search across {len(collection_ids)} collections")
        logger.info(f"📝 Query: '{query}'")
        logger.info(f"📚 Collections: {collection_ids}")

        # Expand query for better retrieval
        expanded_queries = self.expand_query(query)
        logger.info(f"🔄 Expanded queries: {expanded_queries}")

        all_results = []
        successful_collections = 0
        
        for expanded_query in expanded_queries:
            # Embed the query text once per expansion variant instead of once
            # per (expansion x collection) pair — the embedding is identical
            # across collections, only the vector store being searched differs.
            query_vector = self.embeddings.embed_query(expanded_query)
            for collection_id in collection_ids:
                logger.info(f"🔍 Searching collection {collection_id} with query: '{expanded_query}'")
                vector_store = self.get_vector_store(collection_id)
                if vector_store:
                    successful_collections += 1
                    try:
                        # Use similarity_search_with_score_by_vector which returns (doc, distance)
                        # Lower distance = more similar
                        results_with_score = vector_store.similarity_search_with_score_by_vector(
                            query_vector, k=top_k
                        )
                        logger.info(f"📄 Found {len(results_with_score)} results")

                        faiss_ranked = []
                        for doc, distance in results_with_score:
                            # Convert L2 distance to similarity score (0-1 range)
                            similarity_score = 1.0 / (1.0 + float(distance))
                            faiss_ranked.append((doc, similarity_score))

                        # Fuse with lexical BM25 ranking (RRF) so exact-term
                        # matches are not drowned out by dense similarity.
                        bm25, corpus_docs = self.get_bm25_index(collection_id)
                        fused = self._rrf_fuse(
                            expanded_query, faiss_ranked, bm25, corpus_docs, top_k=top_k
                        )

                        for doc, similarity_score in fused:
                            # Accept all results with reasonable similarity (> 0.05)
                            if similarity_score > 0.05:
                                doc.metadata["collection_id"] = collection_id
                                doc.metadata["similarity_score"] = similarity_score
                                doc.metadata["matched_query"] = expanded_query
                                all_results.append((doc, similarity_score))
                                logger.debug(f"✅ Added result with similarity {similarity_score:.4f}")
                            else:
                                logger.debug(f"⏭️ Skipped result with low similarity {similarity_score:.4f}")
                    except Exception as e:
                        logger.error(f"❌ Search failed for {collection_id}: {str(e)}")
                        continue
                else:
                    logger.error(f"❌ Failed to load vector store for {collection_id}")

        logger.info(f"📊 Search summary: {successful_collections}/{len(collection_ids)} collections loaded, {len(all_results)} total results")

        # Remove duplicates and sort by score
        unique_results = {}
        for doc, score in all_results:
            content = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
            content_hash = hash(content[:100])  # Hash first 100 chars
            if content_hash not in unique_results or score > unique_results[content_hash][1]:
                unique_results[content_hash] = (doc, score)

        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        
        import re
        processed_results = []
        for doc, score in sorted_results:
            original_content = doc.page_content
            # Backfill page metadata from [PAGE x] if page is missing/empty/generic
            if "page" not in doc.metadata or doc.metadata["page"] in ["?", "Unknown", None]:
                match = re.search(r'\[PAGE\s+(\d+)\]', original_content)
                if match:
                    doc.metadata["page"] = match.group(1)
            
            # Clean any [PAGE x] markers from content text
            cleaned_content = re.sub(r'\[PAGE\s+\d+\]', '', original_content).strip()
            
            # Skip if the chunk has no useful content remaining
            if not cleaned_content or len(cleaned_content) < 10:
                continue
                
            doc.page_content = cleaned_content
            processed_results.append(doc)
            
        final_results = processed_results[:config.total_k_results]
        
        logger.info(f"🎯 Final results after deduplication: {len(final_results)}")
        return final_results

    def clean_context(self, text: str) -> str:
        """Clean context text to remove noise that confuses the LLM"""
        import re
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove diagram/flowchart artifacts (arrows, boxes)
        text = re.sub(r'[→←↑↓►◄▲▼■□●○◆◇]', '', text)
        text = re.sub(r'[\|│┃┆┊╎]', ' ', text)  # vertical lines
        text = re.sub(r'[-─━]{3,}', ' ', text)  # horizontal lines
        
        # Remove repeated single characters that appear in diagrams
        text = re.sub(r'(\b\w\b\s*){4,}', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\d+\]', '', text)  # footnote markers
        
        # Remove text that looks like reversed/garbled (consecutive uppercase without spaces)
        # Keep meaningful acronyms (2-6 chars) but remove long garbled strings
        text = re.sub(r'\b[A-Z]{7,}\b', '', text)
        
        # Clean up multiple spaces again after removals
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def truncate_context(self, text, max_tokens=500):
        """Truncate context to avoid token limit issues"""
        # Clean the text first
        text = self.clean_context(text)
        
        # Simple word-based truncation
        words = text.split()
        if len(words) > max_tokens:
            truncated = " ".join(words[:max_tokens]) + "... [truncated]"
            logger.warning(f"Context truncated from {len(words)} to {max_tokens} words")
            return truncated
        return text

    def generate_answer(self, context_docs, question):
        """Generate answer with enhanced context handling"""
        if not context_docs:
            return "Maaf, tidak menemukan informasi yang relevan dalam dokumen."

        # Prepare context with source information
        context_parts = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('similarity_score', 0)

            # Truncate each document content to avoid token limit
            truncated_content = self.truncate_context(doc.page_content, max_tokens=150)

            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{truncated_content}\n"
            )

        context = "\n".join(context_parts)

        # Further truncate the entire context if needed
        context = self.truncate_context(context, max_tokens=400)

        # Simplified prompt for flan-t5
        prompt_template = """Answer the question based on the context. Answer in Indonesian.

Context:
{context}

Question: {question}

Answer:"""

        prompt = prompt_template.format(context=context, question=question)

        try:
            result = self.llm.invoke(prompt)
            answer = result.strip()
            
            # Validate output
            if self._is_garbled_output(answer):
                logger.warning(f"Garbled output in generate_answer: {answer[:100]}")
                # Return best document content as fallback
                best_doc = context_docs[0]
                source = best_doc.metadata.get('source', 'dokumen')
                page = best_doc.metadata.get('page', '?')
                content = self.truncate_context(best_doc.page_content, max_tokens=150)
                return f"Berdasarkan {source} (halaman {page}):\n\n{content}"
            
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return "Maaf, terjadi kesalahan dalam menghasilkan jawaban."

    def get_all_collections(self):
        """Return list of available PDF collection IDs (Supabase DB → S3 scan → local disk fallback)."""
        # ── Supabase DB first ───────────────────────────────────────────────
        try:
            import storage as supabase_storage
            if supabase_storage.has_database():
                rows = supabase_storage.list_collections()
                if rows:
                    # Only collections the user has left active are used as a
                    # default source — same "active" concept as public links
                    # and external DB connections.
                    ids = [r["collection_id"] for r in rows if r.get("status", "active") == "active"]
                    logger.info("get_all_collections: %d active of %d from Supabase DB", len(ids), len(rows))
                    return ids
        except Exception as e:
            logger.warning("Supabase list_collections failed: %s", e)

        # ── S3 bucket scan fallback ─────────────────────────────────────────
        try:
            import storage as supabase_storage
            if supabase_storage.is_enabled():
                ids = supabase_storage.list_collection_ids_from_s3()
                if ids:
                    logger.info("get_all_collections: %d from S3 scan", len(ids))
                    return ids
        except Exception as e:
            logger.warning("S3 collection scan failed: %s", e)

        # ── Local disk fallback ────────────────────────────────────────────
        collections = []
        if not os.path.exists(config.index_folder):
            return collections
        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        logger.info("get_all_collections: %d from local disk", len(collections))
        return collections

    def get_all_chat_collections(self):
        """Return list of available chat collection IDs (Supabase DB → S3 scan → local disk)."""
        # ── Supabase DB first ─────────────────────────────────────────────────
        try:
            import storage as supabase_storage
            if supabase_storage.has_database():
                rows = supabase_storage.list_chat_collections()
                if rows:
                    ids = [r["collection_id"] for r in rows if r.get("status", "active") == "active"]
                    logger.info("get_all_chat_collections: %d active of %d from Supabase DB", len(ids), len(rows))
                    return ids
        except Exception as e:
            logger.warning("Supabase list_chat_collections failed: %s", e)

        # ── S3 bucket scan fallback ───────────────────────────────────────────
        try:
            import storage as supabase_storage
            if supabase_storage.is_enabled():
                ids = supabase_storage.list_chat_collection_ids_from_s3()
                if ids:
                    logger.info("get_all_chat_collections: %d from S3 scan", len(ids))
                    return ids
        except Exception as e:
            logger.warning("S3 chat collection scan failed: %s", e)

        # ── Local disk fallback ───────────────────────────────────────────────
        collections = []
        if not os.path.exists(config.chat_index_folder):
            return collections
        for entry in os.listdir(config.chat_index_folder):
            entry_path = os.path.join(config.chat_index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        logger.info("get_all_chat_collections: %d from local disk", len(collections))
        return collections

    def get_chat_vector_store(self, collection_id: str):
        """Get or load chat vector store from cache"""
        cache_key = f"chat_{collection_id}"
        
        with self._cache_lock:
            if cache_key in self.vector_store_cache:
                return self.vector_store_cache[cache_key]
        
        index_path = os.path.join(config.chat_index_folder, collection_id)
        if not os.path.exists(os.path.join(index_path, "index.faiss")):
            try:
                import storage as supabase_storage
                if supabase_storage.is_enabled():
                    logger.info(f"Downloading chat index from S3: {collection_id}")
                    supabase_storage.download_chat_index(collection_id, index_path)
            except Exception as e:
                logger.warning(f"S3 chat index download failed: {e}")
        if not os.path.exists(index_path):
            logger.warning(f"Chat index not found: {index_path}")
            return None
        
        try:
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with self._cache_lock:
                self.vector_store_cache[cache_key] = vector_store
            
            logger.info(f"📱 Loaded chat vector store: {collection_id}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load chat vector store {collection_id}: {e}")
            return None

    def extract_file_reference(self, query: str) -> Optional[str]:
        """Extract file name reference from query"""
        import re
        query_lower = query.lower()
        
        # Pattern: "dari file X", "file X", "di X.txt", "tentang X", etc
        patterns = [
            r'dari file\s+([\w_-]+)',
            r'file\s+([\w_-]+)',
            r'di\s+([\w_-]+\.txt)',
            r'tentang\s+([\w_-]+)',
            r'([\w_-]+\.txt)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                file_ref = match.group(1)
                logger.info(f"📎 Detected file reference in query: {file_ref}")
                return file_ref
        
        return None
    
    def _load_collection_keywords(self, collection_id: str) -> List[str]:
        """Load saved keywords from collection metadata"""
        import json
        import os
        
        metadata_path = os.path.join(config.chat_index_folder, collection_id, "metadata.json")
        
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    keywords = metadata.get('keywords', [])
                    if keywords:
                        logger.debug(f"📚 Loaded {len(keywords)} keywords for {collection_id}")
                        return keywords
        except Exception as e:
            logger.warning(f"Failed to load keywords for {collection_id}: {e}")
        
        return []  # Return empty list if no keywords found
    
    def _expand_chat_query(self, query: str) -> List[str]:
        """Expand query with variations for better chat search"""
        queries = [query]
        query_lower = query.lower()
        
        # Common Indonesian synonyms/variations for chat context
        expansions = {
            'nik': ['nomor induk karyawan', 'employee id', 'id karyawan'],
            'cuti': ['leave', 'libur', 'off', 'saldo cuti'],
            'sisa': ['remaining', 'tersisa', 'balance'],
            'berapa': ['what is', 'jumlah', 'total'],
            'nama': ['name', 'siapa'],
            'email': ['alamat email', 'mail'],
            'telepon': ['phone', 'hp', 'nomor telepon'],
        }
        
        # Add expanded queries
        for term, synonyms in expansions.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # Limit to avoid too many queries
                    expanded = query_lower.replace(term, syn)
                    if expanded not in queries:
                        queries.append(expanded)
        
        return queries[:3]  # Limit to 3 queries max

    def search_across_chat_collections(
        self, 
        question: str, 
        collection_ids: Optional[List[str]] = None,
        file_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List:
        """Search across chat collections with optional file filtering"""
        if collection_ids is None:
            collection_ids = self.get_all_chat_collections()
        
        # Auto-detect file reference if not explicitly provided
        if file_filter is None:
            file_filter = self.extract_file_reference(question)
        
        if not collection_ids:
            logger.info("No chat collections available")
            return []
        
        all_results = []
        
        # Expand query for better search coverage
        search_queries = self._expand_chat_query(question)
        logger.info(f"🔍 Searching with queries: {search_queries}")
        
        # Embed each query variant once and reuse the vector across all
        # collections, instead of re-embedding the same text per collection.
        query_vectors = [(query, self.embeddings.embed_query(query)) for query in search_queries]

        for collection_id in collection_ids:
            vector_store = self.get_chat_vector_store(collection_id)
            if not vector_store:
                continue

            try:
                # Search with multiple query variations
                seen_content_hashes = set()
                relevance_score_fn = vector_store._select_relevance_score_fn()

                for query, query_vector in query_vectors:
                    docs_and_scores = vector_store.similarity_search_with_score_by_vector(
                        query_vector, k=top_k
                    )
                    results = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

                    for doc, score in results:
                        # Deduplicate by content hash
                        content_hash = hash(doc.page_content[:100])
                        if content_hash in seen_content_hashes:
                            continue
                        seen_content_hashes.add(content_hash)
                        
                        source_file = doc.metadata.get('source', '').lower()
                        content_lower = doc.page_content.lower()
                        
                        # Log all results for debugging
                        logger.info(f"🔍 Found: {source_file} with score {score:.3f}")
                        
                        # Apply file filter if specified
                        if file_filter and file_filter.lower() not in source_file:
                            logger.debug(f"⏭️ Skipping {source_file} - doesn't match filter: {file_filter}")
                            continue
                        
                        # Multi-level boosting strategy
                        boosted_score = score
                        boost_reasons = []
                        
                        # 1. Filename matching boost
                        if file_filter and file_filter.lower() in source_file:
                            boosted_score *= 1.8
                            boost_reasons.append("filename_match")
                        
                        # 2. Load saved keywords from metadata (dynamic!)
                        saved_keywords = self._load_collection_keywords(collection_id)
                        
                        # 3. Content keyword matching boost (use saved keywords dynamically)
                        question_lower = question.lower()
                        question_words = set(question_lower.split())
                        
                        # Check for keywords match
                        keyword_matches = 0
                        matched_keywords = []
                        
                        for keyword in saved_keywords:
                            if keyword in question_lower and keyword in content_lower:
                                keyword_matches += 1
                                matched_keywords.append(keyword)
                        
                        # Boost based on keyword density
                        if keyword_matches >= 2:
                            keyword_boost = 1.5
                            boosted_score *= keyword_boost
                            boost_reasons.append(f"{keyword_matches}_keywords[{','.join(matched_keywords[:3])}]")
                            logger.info(f"🎯 Keyword boost for {source_file}: {keyword_matches} matches ({matched_keywords[:5]})")
                        
                        if boost_reasons:
                            logger.info(f"⬆️ Boosted score for {source_file}: {score:.3f} → {boosted_score:.3f} ({', '.join(boost_reasons)})")
                        
                        # Use VERY low threshold since FAISS relevance scores can be negative
                        # The key is to compare relative scores, not absolute values
                        threshold = 0.0 if file_filter else 0.05  # Allow negative scores with filtering
                        
                        if boosted_score >= threshold or (file_filter and file_filter.lower() in source_file):
                            # If file filter matches, always include regardless of score
                            doc.metadata['similarity_score'] = float(boosted_score)
                            doc.metadata['original_score'] = float(score)
                            doc.metadata['collection_id'] = collection_id
                            all_results.append(doc)
                            logger.info(f"✅ Added {source_file} (score: {boosted_score:.3f})")
                        else:
                            logger.debug(f"⏭️ Skipped {source_file} with score {boosted_score:.3f} < {threshold}")
                        
            except Exception as e:
                logger.error(f"Error searching chat collection {collection_id}: {e}")
                continue
        
        # Re-rank results based on keyword content match with query
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for doc in all_results:
            content_lower = doc.page_content.lower()
            
            # Count how many query words appear in content
            word_matches = sum(1 for word in question_words if word in content_lower and len(word) > 2)
            
            # Boost score based on keyword overlap
            current_score = doc.metadata.get('similarity_score', 0)
            content_boost = 1 + (word_matches * 0.2)  # 20% boost per matching word
            doc.metadata['similarity_score'] = current_score * content_boost
            doc.metadata['keyword_matches'] = word_matches
            
            if word_matches > 0:
                logger.info(f"📝 Content re-rank for {doc.metadata.get('source', '')}: {word_matches} word matches, score: {current_score:.3f} → {doc.metadata['similarity_score']:.3f}")
        
        # Sort by updated score
        all_results.sort(key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)
        
        filter_info = f" (filtered by: {file_filter})" if file_filter else ""
        logger.info(f"📱 Found {len(all_results)} chat results{filter_info}")
        
        if file_filter and len(all_results) == 0:
            logger.warning(f"⚠️ No results found with file filter '{file_filter}'. Try broader search.")
        
        return all_results[:top_k * 2]  # Return more context for chats

    def _extract_google_drive_file_id(self, raw_url: str) -> Optional[str]:
        parsed = urlparse(raw_url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host not in {"drive.google.com", "docs.google.com"}:
            return None

        match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
        if match:
            return match.group(1)

        query = parse_qs(parsed.query)
        return query.get("id", [None])[0]

    def _normalize_public_link_download_url(self, raw_url: str) -> str:
        file_id = self._extract_google_drive_file_id(raw_url)
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return raw_url

    def _extract_remote_filename(self, response: Any, source_url: str, fallback_name: str) -> str:
        content_disposition = response.headers.get("content-disposition", "")
        filename_star = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if filename_star:
            return unquote(filename_star.group(1).strip())

        filename = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
        if filename:
            return filename.group(1).strip()

        parsed = urlparse(str(response.url or source_url))
        candidate = os.path.basename(parsed.path.rstrip("/"))
        return candidate or fallback_name

    def _download_public_link_item(self, item: Dict[str, Any], destination_dir: str) -> Optional[Tuple[str, str]]:
        import requests
        from ssrf_guard import assert_public_url_safe

        source_url = item.get("url", "").strip()
        if not source_url:
            return None

        download_url = self._normalize_public_link_download_url(source_url)
        fallback_name = item.get("name") or f"public-link-{uuid.uuid4().hex[:8]}"

        try:
            assert_public_url_safe(download_url)
        except Exception as exc:
            logger.warning("Rejected public link item %s: %s", source_url, exc)
            return None

        try:
            response = requests.get(
                download_url,
                stream=True,
                timeout=(20, 60),
                allow_redirects=True,
                headers={"User-Agent": "DocuLens/1.0 (+public-link-runtime)"},
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed downloading public link item %s: %s", source_url, exc)
            return None

        content_type = (response.headers.get("content-type") or "").lower()
        resolved_name = self._extract_remote_filename(response, source_url, fallback_name)
        suffix = Path(resolved_name).suffix.lower()

        if "pdf" in content_type or suffix == ".pdf":
            final_name = resolved_name if resolved_name.lower().endswith(".pdf") else f"{resolved_name}.pdf"
        elif content_type.startswith("text/") or suffix in {".txt", ".md", ".log"}:
            final_name = resolved_name if suffix in {".txt", ".md", ".log"} else f"{resolved_name}.txt"
        else:
            logger.info("Skipping unsupported public link item %s (content-type=%s)", source_url, content_type)
            return None

        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(final_name).name).strip(" ._") or fallback_name
        destination_path = os.path.join(destination_dir, safe_name)

        try:
            with open(destination_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
        finally:
            response.close()

        return destination_path, safe_name

    def _load_public_link_documents(self, public_link_sources: List[Dict[str, Any]]) -> List[Document]:
        documents: List[Document] = []
        seen_urls = set()

        with tempfile.TemporaryDirectory(prefix="public-link-runtime-") as temp_dir:
            for source in public_link_sources:
                link_id = source.get("link_id", "")
                link_title = source.get("title") or link_id or "Public Link"
                items = source.get("items") or []
                if not items and source.get("url"):
                    items = [{"url": source["url"], "name": link_title, "item_type": "file"}]

                for item in items:
                    item_url = (item.get("url") or "").strip()
                    if not item_url or item_url in seen_urls or item.get("item_type") == "folder":
                        continue
                    seen_urls.add(item_url)

                    downloaded = self._download_public_link_item(item, temp_dir)
                    if not downloaded:
                        continue

                    file_path, file_name = downloaded
                    extension = Path(file_path).suffix.lower()

                    try:
                        if extension == ".pdf":
                            pdf_loader = PyPDFLoader(file_path)
                            loaded_docs = pdf_loader.load()
                        elif extension in {".txt", ".md", ".log"}:
                            try:
                                text_loader = TextLoader(file_path, encoding="utf-8")
                                loaded_docs = text_loader.load()
                            except UnicodeDecodeError:
                                text_loader = TextLoader(file_path, encoding="latin-1")
                                loaded_docs = text_loader.load()
                        else:
                            continue
                    except Exception as exc:
                        logger.warning("Failed loading public link item %s: %s", item_url, exc)
                        continue

                    # Merge all pages of one file into a single Document before
                    # splitting. Slide-deck PDFs carry only a few words per page,
                    # so page-level chunks end up as bare headings with no body
                    # text for the LLM to work with.
                    merged_text = "\n\n".join(
                        doc.page_content for doc in loaded_docs if doc.page_content.strip()
                    )
                    if not merged_text.strip():
                        continue

                    merged_doc = Document(
                        page_content=merged_text,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,
                            "source_kind": "public_link",
                            "public_link_id": link_id,
                            "public_link_title": link_title,
                            "public_link_url": source.get("url"),
                            "item_url": item_url,
                            "collection_id": f"public-link:{link_id}" if link_id else "public-link",
                        },
                    )
                    documents.append(merged_doc)

        return documents

    # ── Hybrid sparse+dense retrieval helpers (BM25 + FAISS via RRF) ────────

    def _bm25_tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _build_bm25(self, docs: List[Document]):
        """Build a BM25 index over documents; returns None when unavailable."""
        if not docs:
            return None
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed; falling back to dense-only retrieval")
            return None
        corpus = [self._bm25_tokenize(d.page_content) for d in docs]
        if not any(corpus):
            return None
        try:
            return BM25Okapi(corpus)
        except Exception as exc:
            logger.warning("BM25 index build failed: %s", exc)
            return None

    def _rrf_fuse(
        self,
        query: str,
        faiss_ranked: List[Tuple[Document, float]],
        bm25,
        corpus_docs: List[Document],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """Fuse FAISS and BM25 rankings with Reciprocal Rank Fusion.

        Returns (doc, display_similarity) pairs ordered by fused score. The
        display similarity stays FAISS-based so UI "% match" keeps its meaning;
        BM25-only docs get a conservative floor value.
        """
        scores: Dict[int, float] = {}
        doc_by_key: Dict[int, Tuple[Document, float]] = {}

        def key_of(d: Document) -> int:
            return hash(d.page_content[:200])

        for rank, (doc, sim) in enumerate(faiss_ranked):
            k = key_of(doc)
            doc_by_key.setdefault(k, (doc, sim))
            scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)

        bm25_top_keys: List[int] = []
        if bm25 is not None and corpus_docs:
            tokens = self._bm25_tokenize(query)
            if tokens:
                bm25_scores = bm25.get_scores(tokens)
                order = sorted(
                    range(len(corpus_docs)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )
                floor_sim = max(
                    min((s for _, s in faiss_ranked), default=0.1), 0.1
                )
                for rank, idx in enumerate(order[: max(top_k * 3, 10)]):
                    if bm25_scores[idx] <= 0:
                        break
                    d = corpus_docs[idx]
                    k = key_of(d)
                    doc_by_key.setdefault(k, (d, floor_sim))
                    scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)
                    if rank < 2:
                        bm25_top_keys.append(k)

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        # Per-file diversity cap: without it a couple of large documents can
        # occupy every slot and crowd out the file that actually answers the
        # question.
        max_per_source = 2
        selected: List[int] = []
        overflow: List[int] = []
        per_source: Dict[Any, int] = {}
        for k, _ in fused:
            src = doc_by_key[k][0].metadata.get("source")
            if per_source.get(src, 0) < max_per_source:
                selected.append(k)
                per_source[src] = per_source.get(src, 0) + 1
            else:
                overflow.append(k)
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            selected.extend(overflow[: top_k - len(selected)])

        # Guaranteed lexical slot: plain RRF favors docs present in BOTH
        # rankings, which can push the strongest exact-keyword match out of a
        # small top_k. Keep BM25's single best hit in the result (one slot
        # only — two slots crowded out dense hits when top_k is small).
        guaranteed = [k for k in bm25_top_keys[:1] if k not in selected]
        if guaranteed and top_k > len(guaranteed):
            keep = [k for k in selected if k not in guaranteed]
            selected = (keep[: top_k - len(guaranteed)] + guaranteed)[:top_k]

        return [doc_by_key[k] for k in selected]

    def get_bm25_index(self, collection_id: str):
        """Lazily build and cache a BM25 index over an indexed collection's chunks."""
        with self._cache_lock:
            if collection_id in self.bm25_cache:
                return self.bm25_cache[collection_id]
            vector_store = self.vector_store_cache.get(collection_id)
            if vector_store is None:
                return None, []
            try:
                docs = list(vector_store.docstore._dict.values())
            except Exception as exc:
                logger.warning("BM25: could not read docstore for %s: %s", collection_id, exc)
                return None, []
            entry = (self._build_bm25(docs), docs)
            self.bm25_cache[collection_id] = entry
            return entry

    def search_public_links_realtime(
        self,
        query: str,
        public_link_sources: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Document]:
        if not public_link_sources:
            return []

        documents = self._load_public_link_documents(public_link_sources)
        if not documents:
            logger.info("No readable realtime documents found for public links")
            return []

        # Larger chunks than the global config: public-link sources are often
        # slide decks whose per-page text is tiny, so chunks must span several
        # pages to give the LLM usable context.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.public_link_chunk_size,
            chunk_overlap=config.public_link_chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            return []

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        # Wider FAISS candidate pool than top_k so RRF fusion with BM25 has
        # dense scores for (nearly) every lexical match it promotes.
        k_faiss = min(len(chunks), max(top_k * 3, 10))
        results_with_score = vector_store.similarity_search_with_score(query, k=k_faiss)

        faiss_ranked: List[Tuple[Document, float]] = []
        for doc, distance in results_with_score:
            faiss_ranked.append((doc, 1.0 / (1.0 + float(distance))))

        bm25 = self._build_bm25(chunks)
        fused = self._rrf_fuse(query, faiss_ranked, bm25, chunks, top_k=max(top_k, 1))

        realtime_results: List[Document] = []
        for doc, similarity_score in fused:
            if similarity_score <= 0.05:
                continue
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata.setdefault("source_kind", "public_link")
            realtime_results.append(doc)

        return realtime_results[: config.total_k_results]

    def invalidate_cache(self, collection_id=None):
        """Invalidate cache for specific collection or all"""
        with self._cache_lock:
            if collection_id:
                if collection_id in self.vector_store_cache:
                    del self.vector_store_cache[collection_id]
                if collection_id in self.bm25_cache:
                    del self.bm25_cache[collection_id]
                # Also check chat cache
                chat_key = f"chat_{collection_id}"
                if chat_key in self.vector_store_cache:
                    del self.vector_store_cache[chat_key]
            else:
                self.vector_store_cache.clear()
                self.bm25_cache.clear()

    def initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_manager = db_manager
            self._db_initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db_initialized = False

    def get_llm(self, provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[Any, str]:
        """
        Get LLM instance based on provider and model.
        Returns (llm_instance, model_identifier_string)
        
        If provider/model not specified, uses config defaults.
        Caches LLM instances for reuse.
        """
        # Determine provider/model defaults for request without explicit override
        if not provider and not model:
            llm_provider = config.default_llm_provider
            llm_model = config.default_llm_model
        elif provider:
            try:
                llm_provider = LLMProvider(provider.lower())
            except ValueError:
                logger.warning(f"Invalid provider '{provider}', using default")
                llm_provider = config.default_llm_provider
            llm_model = model or (config.gemini_model if llm_provider == LLMProvider.GEMINI else config.model_name)
        else:
            inferred_gemini = bool(model and model.lower().startswith("gemini"))
            llm_provider = LLMProvider.GEMINI if inferred_gemini else config.default_llm_provider
            llm_model = model or config.default_llm_model
        
        # Create cache key
        cache_key = f"{llm_provider.value}:{llm_model}"
        model_identifier = f"{llm_provider.value}/{llm_model}"
        
        # Check cache
        if cache_key in self._llm_cache:
            logger.info(f"🔄 Using cached LLM: {model_identifier}")
            return self._llm_cache[cache_key], model_identifier
        
        # Load new LLM
        logger.info(f"🚀 Loading LLM: {model_identifier}")
        
        try:
            if llm_provider == LLMProvider.HUGGINGFACE:
                llm = self._load_huggingface_llm(llm_model)
            elif llm_provider == LLMProvider.GEMINI:
                llm = self._load_gemini_llm(llm_model)
            else:
                raise ValueError(f"Unsupported provider: {llm_provider}")
            
            # Cache the LLM
            self._llm_cache[cache_key] = llm
            self._current_provider = llm_provider
            self._current_model = llm_model
            
            logger.info(f"✅ LLM loaded successfully: {model_identifier}")
            return llm, model_identifier
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_identifier}: {e}")
            if provider or model:
                logger.info("⚠️ Falling back to configured default LLM")
                return self.get_llm(config.default_llm_provider.value, config.default_llm_model)
            raise
    
    def _load_huggingface_llm(self, model_name: str):
        """Load HuggingFace model (local)"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            logger.warning(f"Model {model_name} failed, using fallback")
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            batch_size=4 if torch.cuda.is_available() else 1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def _load_gemini_llm(self, model_name: str):
        """Load Google Gemini model (cloud - free tier)"""
        if not config.gemini_api_key:
            logger.error("❌ GEMINI_API_KEY not set in .env file")
            raise ValueError("GEMINI_API_KEY not configured. Please add it to .env file or use HuggingFace/Ollama")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.error("❌ langchain-google-genai not installed")
            raise ImportError("Please install: pip install langchain-google-genai")
        
        logger.info(f"🔑 Using Gemini API with model: {model_name}")
        
        # Fix model name - Gemini uses different naming convention
        # Map common names to actual Gemini model names
        model_mapping = {
            'gemini-1.5-flash': 'gemini-1.5-flash-latest',
            'gemini-1.5-pro': 'gemini-1.5-pro-latest',
            'gemini-pro': 'gemini-1.5-pro-latest',
        }
        
        actual_model = model_mapping.get(model_name, model_name)
        if actual_model != model_name:
            logger.info(f"🔄 Model name mapped: {model_name} -> {actual_model}")
        
        return ChatGoogleGenerativeAI(
            model=actual_model,
            google_api_key=config.gemini_api_key,
            temperature=config.temperature,
        )
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Return available models per provider"""
        return {provider.value: models for provider, models in AVAILABLE_MODELS.items()}
    
    def get_current_model_info(self) -> str:
        """Get current model identifier string"""
        if self._current_provider and self._current_model:
            return f"{self._current_provider.value}/{self._current_model}"
        return f"{config.default_llm_provider.value}/{config.default_llm_model}"

    def initialize_components(self):
        """Initialize all components including database"""
        with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing NLP components...")
            try:
                # Initialize embeddings first (faster to load)
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.embedding_model,
                    model_kwargs={
                        'device': 'cuda' if torch.cuda.is_available()
                        else 'cpu'
                    },
                    encode_kwargs={'normalize_embeddings': True}
                )

                # Load default LLM
                self.llm, model_id = self.get_llm()
                
                # Keep tokenizer reference for HuggingFace models
                if config.default_llm_provider == LLMProvider.HUGGINGFACE:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                    except:
                        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

                self.initialize_database()
            
                self._initialized = True
                logger.info(f"✅ All components initialized with model: {model_id}")

            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                raise

    def expand_query_for_db(self, query: str) -> List[str]:
        """Expand query with database-related terms"""
        expanded_queries = [query]

        # Add lowercase version
        expanded_queries.append(query.lower())

        # Add synonyms for database-related terms
        for term, synonyms in self.db_query_expansion_terms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(term, synonym)
                    expanded_queries.append(expanded_query)

        return list(set(expanded_queries))

    def analyze_question_type(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine optimal search strategy with smart table routing"""
        question_lower = question.lower()

        #Database-related keywords
        db_keywords = [
            'user', 'profile', 'customer', 'product', 'order', 'price', 
            'jumlah', 'total', 'data', 'tabel', 'table', 'database', 'sql',
            'nama', 'email', 'alamat', 'tanggal', 'date', 'harga', 'stock',
            'karyawan', 'transaksi', 'pesanan', 'siapa', 'pegawai', 'staff',
            'anggota', 'member', 'pelanggan', 'department', 'departemen'
        ]

        # Use pdf_keywords from instance (includes LPDU document terms)
        pdf_terms = getattr(self, 'pdf_keywords', [
            'dokumen', 'pdf', 'file', 'laporan', 'report', 'handbook',
            'kebijakan', 'policy', 'prosedur', 'pedoman', 'guideline',
            'kontrak', 'agreement', 'proposal'
        ])

        is_db_question = any(keyword in question_lower for keyword in db_keywords)
        is_pdf_question = any(keyword in question_lower for keyword in pdf_terms)

        # Smart table routing - determine which tables to search
        target_tables = self.get_target_tables(question_lower)
        
        # Log detection results
        logger.info(f"📊 Question analysis: is_db={is_db_question}, is_pdf={is_pdf_question}")

        # if both or unclear, use hybrid
        if (is_db_question and is_pdf_question) or (not is_db_question and not is_pdf_question):
            recommended_type = SearchType.HYBRID
            logger.info(f"🔄 Using HYBRID search (both sources)")
        elif is_db_question:
            recommended_type = SearchType.STRUCTURED
            logger.info(f"🗄️ Using STRUCTURED search (database)")
        else:
            recommended_type = SearchType.UNSTRUCTURED
            logger.info(f"📄 Using UNSTRUCTURED search (PDF)")

        return {
            "recommended_type": recommended_type,
            "is_db_question": is_db_question,
            "is_pdf_question": is_pdf_question,
            "search_terms": self.extract_search_terms(question),
            "target_tables": target_tables  # NEW: specific tables to search
        }

    def get_target_tables(self, question_lower: str) -> List[str]:
        """Determine which database tables to search based on question content"""
        target_tables = []
        table_scores: Dict[str, int] = {}
        
        # Check for person-related questions (routes to user_profiles)
        is_person_question = any(pattern in question_lower for pattern in self.person_question_patterns)
        if is_person_question:
            table_scores["user_profiles"] = table_scores.get("user_profiles", 0) + 2  # Higher weight
        
        # Score based on keywords
        for table_name, keywords in self.table_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                table_scores[table_name] = table_scores.get(table_name, 0) + score
        
        if table_scores:
            # Sort by score and return tables with matches
            sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            target_tables = [table for table, score in sorted_tables]
            logger.info(f"Smart routing: targeting tables {target_tables} (scores: {table_scores})")
        else:
            # No specific match, search all tables (fallback)
            target_tables = list(self.table_keywords.keys())
            logger.info(f"No specific table match, searching all: {target_tables}")
        
        return target_tables

    # def extract_search_terms(self, question: str) -> List[str]:
    #     """Extract meaningful search terms from question - NO hardcoded values"""
    #     import re
        
    #     # Stop words to filter out (common question words, not search-worthy)
    #     stop_words = {
    #         'apa', 'siapa', 'dimana', 'kapan', 'berapa', 'bagaimana', 'mengapa',
    #         'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'dalam', 'pada', 'untuk',
    #         'adalah', 'ini', 'itu', 'dengan', 'seperti', 'jika', 'maka',
    #         'cari', 'tampilkan', 'semua', 'lihat', 'tunjukkan', 'show',
    #         'find', 'search', 'get', 'the', 'what', 'who', 'where', 'when', 'how',
    #         'jumlah', 'total', 'hitung', 'count', 'berapa', 'banyak', 'orang'  # aggregation + generic words
    #     }
        
    #     # Column/field words that indicate we're looking for a value, not searching
    #     field_indicators = {'department', 'departemen', 'divisi', 'bagian', 'posisi', 'jabatan'}
        
    #     # Remove punctuation but keep alphanumeric and spaces
    #     cleaned = re.sub(r'[^\w\s]', ' ', question)
    #     words = cleaned.split()
        
    #     logger.info(f"🔍 Words from query: {words}")
        
    #     meaningful_terms = []
        
    #     for i, word in enumerate(words):
    #         word_lower = word.lower().strip()
            
    #         # Skip stop words first
    #         if word_lower in stop_words:
    #             continue
            
    #         # If previous word was a field indicator, this word is likely a VALUE
    #         # Keep original case for proper nouns like "IT", "HR", "Finance"
    #         if i > 0 and words[i-1].lower() in field_indicators:
    #             # This is likely a value like "IT", "HR", "Finance"
    #             logger.info(f"🔍 Found value after field indicator: {word}")
    #             meaningful_terms.append(word.strip())  # Keep original case
    #             continue
            
    #         # Skip field indicators themselves
    #         if word_lower in field_indicators:
    #             continue
            
    #         # For short words (2 chars), only keep if they look like acronyms (all caps)
    #         if len(word_lower) <= 2:
    #             if word.isupper() and len(word) >= 2:
    #                 logger.info(f"🔍 Keeping acronym: {word}")
    #                 meaningful_terms.append(word)  # Keep "IT", "HR", etc.
    #             continue
                
    #         # Add other meaningful terms
    #         meaningful_terms.append(word_lower)
        
    #     unique_terms = list(set(meaningful_terms))
        
    #     logger.info(f"🔍 Extracted search terms: {unique_terms}")
    #     return unique_terms

    # processor.py - perbaiki extract_search_terms
    def extract_search_terms(self, question: str) -> Dict[str, Any]:
        """Enhanced search terms extraction dengan entity recognition"""
        import re
        
        # Entity patterns - UPDATED to include more software/brand names
        patterns = {
            'product_names': r'\b(Laptop|Smartphone|Chair|Software|Mouse|ThinkPad|Galaxy|Ergonomic|Project|JetBrains|SAP|Jira|Confluence|AWS|Azure|Datadog|SonarQube|GitLab|Jenkins|Docker|Kubernetes)\b',
            'departments': r'\b(IT|HR|Finance|Marketing|Sales|Operations)\b',
            'positions': r'\b(Engineer|Manager|Analyst|Specialist|Administrator|Director)\b',
            'currencies': r'Rp\s*\d+[.,]\d+|\d+\s*(juta|ribu)',
            'numbers': r'\b\d+\b',
            'dates': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        }
        
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Extract keywords (improved version)
        stop_words = {'apa', 'siapa', 'dimana', 'kapan', 'berapa', 'bagaimana', 
                    'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'dalam', 'untuk'}
        
        question_lower = question.lower()
        words = re.findall(r'\b[\w+]+\b', question_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Detect if query contains specific field references
        field_refs = []
        field_mapping = {
            'harga': 'price', 'nama': 'name', 'email': 'email', 
            'departemen': 'department', 'posisi': 'position',
            'jumlah': 'quantity', 'stok': 'stock_quantity', 'status': 'status'
        }
        
        for idn_field, eng_field in field_mapping.items():
            if idn_field in question_lower:
                field_refs.append(eng_field)
        
        return {
            "keywords": list(set(keywords)),
            "entities": entities,
            "field_references": field_refs,
            "original_terms": words
        }

    def query_structured_data(self, search_terms: List[str], target_tables: Optional[List[str]] = None) -> Dict[str, DatabaseResult]:
        """Query structured data from database with query expansion and smart routing"""
        logger.info(f"📊 query_structured_data called with terms: {search_terms}, tables: {target_tables}")
        logger.info(f"📊 _db_initialized: {self._db_initialized}")
        
        if not self._db_initialized:
            logger.warning("⚠️ Database not initialized, returning empty results")
            return {}

        try:
            # Apply query expansion to search terms
            expanded_terms = self.expand_search_terms_for_db(search_terms)
            logger.info(f"🔍 Original terms: {search_terms} -> Expanded: {expanded_terms}")
            
            # Use target tables if provided, otherwise search all
            tables_to_search = target_tables if target_tables else config.db_tables
            
            db_results = self.db_manager.search_in_specific_tables(
                expanded_terms, 
                tables_to_search,
                limit=config.db_result_limit
            )

            formatted_results = {}
            for table_name, records in db_results.items():
                formatted_results[table_name] = DatabaseResult(
                    table=table_name,
                    data=records,
                    record_count=len(records)
                )

            return formatted_results
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return {}

    def expand_search_terms_for_db(self, search_terms: List[str]) -> List[str]:
        """Expand search terms with synonyms and stemming for better DB matches"""
        expanded = set(search_terms)
        
        # Brand/product names that should NOT be stemmed
        brand_whitelist = {
            'jetbrains', 'jira', 'confluence', 'aws', 'azure', 'datadog', 
            'sonarqube', 'gitlab', 'jenkins', 'docker', 'kubernetes', 'sap'
        }
        
        for term in search_terms:
            term_lower = term.lower()
            
            # Add synonyms from expansion dictionary
            for key, synonyms in self.db_query_expansion_terms.items():
                if key in term_lower or term_lower in synonyms:
                    expanded.add(key)
                    expanded.update(synonyms)
            
            # Apply simple Indonesian stemming ONLY if not a brand name
            if term_lower not in brand_whitelist:
                stemmed = self.simple_indonesian_stem(term_lower)
                if stemmed != term_lower:
                    expanded.add(stemmed)
        
        return list(expanded)

    def simple_indonesian_stem(self, word: str) -> str:
        """Simple Indonesian stemming - remove common affixes"""
        word = word.lower().strip()
        
        # Common Indonesian suffixes
        suffixes = ['kan', 'an', 'i', 'nya', 'lah', 'kah']
        prefixes = ['me', 'di', 'ke', 'se', 'ber', 'ter', 'pe']
        
        # Remove suffixes first
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        # Then remove prefixes
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        return word

    # def hybrid_search(
    #     self, 
    #     question: str, 
    #     collection_ids: Optional[List[str]] = None,
    #     include_chat: bool = True,
    #     include_pdf: bool = True,
    #     include_db: bool = True,
    #     chat_collection_ids: Optional[List[str]] = None
    # ) -> Dict[str, Any]:
    #     """Perform hybrid search across PDF, database, and chat data with collection selection"""
        
    #     logger.info(f"🔍 Hybrid search flags - PDF: {include_pdf}, DB: {include_db}, Chat: {include_chat}")

    #     analysis = self.analyze_question_type(question)
    #     search_terms = analysis["search_terms"]
    #     target_tables = analysis.get("target_tables", [])  # Get smart-routed tables

    #     pdf_docs = []
    #     # FIXED: When user explicitly enables PDF search via include_pdf=True,
    #     # always search PDFs regardless of question type analysis
    #     if include_pdf:
    #         logger.info("📄 Searching PDF collections (explicitly enabled)...")
    #         pdf_docs = self.search_across_collections(
    #             question,
    #             collection_ids=collection_ids,
    #             top_k=config.k_per_collection
    #         )
    #     else:
    #         logger.info("⏭️ PDF search skipped (disabled by user)")

    #     db_results = {}
    #     # FIXED: When user explicitly enables DB search, always search DB
    #     if include_db:
    #         logger.info("🗄️ Searching database...")
    #         # Pass target_tables for smart routing
    #         db_results = self.query_structured_data(search_terms, target_tables)
    #     else:
    #         logger.info("⏭️ Database search skipped (disabled or not relevant)")

    #     # Search chat collections with collection selection
    #     chat_docs = []
    #     if include_chat:
    #         logger.info("💬 Searching chat collections...")
    #         # Auto-detect file reference from question
    #         file_filter = self.extract_file_reference(question)
    #         if file_filter:
    #             logger.info(f"🎯 File filter detected: {file_filter}")
            
    #         chat_docs = self.search_across_chat_collections(
    #             question,
    #             collection_ids=chat_collection_ids,  # Use specific chat collections
    #             file_filter=file_filter,
    #             top_k=config.k_per_collection
    #         )
    #     else:
    #         logger.info("⏭️ Chat search skipped (disabled)")

    #     return {
    #         "pdf_documents": pdf_docs,
    #         "database_results": db_results,
    #         "chat_documents": chat_docs,  # NEW
    #         "search_analysis": analysis,
    #         "search_terms": search_terms,
    #         "target_tables": target_tables  # Include for debugging/transparency
    #     }

    # def hybrid_search(
    #     self, 
    #     question: str, 
    #     collection_ids: Optional[List[str]] = None,
    #     include_chat: bool = True,
    #     include_pdf: bool = True,
    #     include_db: bool = True,
    #     chat_collection_ids: Optional[List[str]] = None
    # ) -> Dict[str, Any]:
    #     """Perform hybrid search across PDF, database, and chat data with collection selection"""
        
    #     logger.info(f"🔍 Hybrid search flags - PDF: {include_pdf}, DB: {include_db}, Chat: {include_chat}")

    #     analysis = self.analyze_question_type(question)
    #     search_terms = analysis["search_terms"]
    #     target_tables = analysis.get("target_tables", [])  # Get smart-routed tables

    #     pdf_docs = []
    #     # FIXED: When user explicitly enables PDF search via include_pdf=True,
    #     # always search PDFs regardless of question type analysis
    #     if include_pdf:
    #         logger.info("📄 Searching PDF collections (explicitly enabled)...")
    #         pdf_docs = self.search_across_collections(
    #             question,
    #             collection_ids=collection_ids,
    #             top_k=config.k_per_collection
    #         )
    #     else:
    #         logger.info("⏭️ PDF search skipped (disabled by user)")

    #     db_results = {}
    #     # FIXED: When user explicitly enables DB search, always search DB
    #     if include_db:
    #         logger.info("🗄️ Searching database...")
    #         # Pass target_tables for smart routing
    #         db_results = self.query_structured_data(search_terms, target_tables)
    #     else:
    #         logger.info("⏭️ Database search skipped (disabled or not relevant)")

    #     # Search chat collections with collection selection
    #     chat_docs = []
    #     if include_chat:
    #         logger.info("💬 Searching chat collections...")
    #         # Auto-detect file reference from question
    #         file_filter = self.extract_file_reference(question)
    #         if file_filter:
    #             logger.info(f"🎯 File filter detected: {file_filter}")
            
    #         chat_docs = self.search_across_chat_collections(
    #             question,
    #             collection_ids=chat_collection_ids,  # Use specific chat collections
    #             file_filter=file_filter,
    #             top_k=config.k_per_collection
    #         )
    #     else:
    #         logger.info("⏭️ Chat search skipped (disabled)")

    #     # Gabungkan hasil dari semua sumber dan beri peringkat berdasarkan skor relevansi
    #     combined_results = self.rank_and_combine_results(
    #         question, 
    #         pdf_docs, 
    #         db_results, 
    #         chat_docs, 
    #         search_terms
    #     )

    #     return {
    #         "pdf_documents": pdf_docs,
    #         "database_results": db_results,
    #         "chat_documents": chat_docs,  # NEW
    #         "search_analysis": analysis,
    #         "search_terms": search_terms,
    #         "target_tables": target_tables,  # Include for debugging/transparency
    #         "combined_results": combined_results  # Hasil gabungan yang sudah di-ranking
    #     }
    
    # ── External (user-connected) database retrieval ────────────────────────
    # This is intentionally separate from self.db_manager / query_structured_data
    # above: that path is a single hardcoded connection used for THIS app's own
    # data. This path opens a fresh connection per active connection the user
    # set up in Sources > Database, schema-agnostic (no ALLOWED_TABLES
    # whitelist) — mirrors PostgreSQLAdapter's "all tables" mode from the
    # research notebook.

    def _serialize_table_rows(self, table_name: str, columns: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
        # Markdown table, not "col: value | col: value" prose lines. LLMs are
        # heavily trained on markdown tables and parse/attend to them far
        # more reliably than ad-hoc key-value dumps — verified empirically:
        # the pipe-delimited format put the exact right row in context and
        # the model still answered "not found".
        col_names = [c["name"] for c in columns]

        def cell(value: Any) -> str:
            text = "" if value is None else str(value)
            return text.replace("|", "\\|").replace("\n", " ")

        lines = [
            f"Tabel: {table_name}",
            "",
            "| " + " | ".join(col_names) + " |",
            "|" + "|".join(["---"] * len(col_names)) + "|",
        ]
        for row in rows:
            lines.append("| " + " | ".join(cell(row.get(k)) for k in col_names) + " |")
        return "\n".join(lines)

    def _load_external_db_documents(self, db_connections: List[Dict[str, Any]]) -> List[Document]:
        from router.database_connections import open_external_connection, list_tables_with_columns

        documents: List[Document] = []

        for conn_info in db_connections:
            connection_id = conn_info.get("connection_id", "")
            label = conn_info.get("label") or connection_id or "Database"
            url = conn_info.get("url", "")
            if not url:
                continue

            try:
                ext_conn = open_external_connection(url)
            except Exception as exc:
                logger.warning("External DB connect failed for %s: %s", label, exc)
                continue

            try:
                tables = list_tables_with_columns(ext_conn, max_tables=config.external_db_max_tables)
                for table in tables:
                    table_name = table["name"]
                    columns = table.get("columns") or []
                    if not columns:
                        continue
                    safe_name = '"' + table_name.replace('"', '""') + '"'
                    try:
                        with ext_conn.cursor() as cur:
                            cur.execute(
                                f"SELECT * FROM {safe_name} LIMIT %s",
                                (config.external_db_max_rows_per_table,),
                            )
                            rows = cur.fetchall() or []
                    except Exception as exc:
                        logger.warning("External DB table read failed (%s.%s): %s", label, table_name, exc)
                        continue

                    if not rows:
                        continue

                    text = self._serialize_table_rows(table_name, columns, rows)
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": table_name,
                            "source_kind": "external_db",
                            "external_db_connection_id": connection_id,
                            "external_db_label": label,
                            "collection_id": f"external-db:{connection_id}" if connection_id else "external-db",
                            "row_count": len(rows),
                        },
                    ))
            finally:
                ext_conn.close()

        return documents

    def search_external_db_realtime(
        self,
        query: str,
        db_connections: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Document]:
        if not db_connections:
            return []

        documents = self._load_external_db_documents(db_connections)
        if not documents:
            logger.info("No readable tables found for active database connections")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.external_db_chunk_size,
            chunk_overlap=config.external_db_chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            return []

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        k_faiss = min(len(chunks), max(top_k * 3, 10))
        results_with_score = vector_store.similarity_search_with_score(query, k=k_faiss)

        faiss_ranked: List[Tuple[Document, float]] = []
        for doc, distance in results_with_score:
            faiss_ranked.append((doc, 1.0 / (1.0 + float(distance))))

        bm25 = self._build_bm25(chunks)
        fused = self._rrf_fuse(query, faiss_ranked, bm25, chunks, top_k=max(top_k, 1))

        realtime_results: List[Document] = []
        for doc, similarity_score in fused:
            if similarity_score <= 0.05:
                continue
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata.setdefault("source_kind", "external_db")
            realtime_results.append(doc)

        return realtime_results[: config.total_k_results]

    # processor.py - perbaiki hybrid_search
    def hybrid_search(
        self,
        question: str,
        collection_ids: Optional[List[str]] = None,
        include_chat: bool = True,
        include_pdf: bool = True,
        include_db: bool = True,
        chat_collection_ids: Optional[List[str]] = None,
        include_public_links: bool = False,
        public_link_sources: Optional[List[Dict[str, Any]]] = None,
        include_external_db: bool = False,
        external_db_connections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Enhanced hybrid search dengan context-aware aggregation"""
        
        # Initialize database if needed
        if include_db and not self._db_initialized:
            logger.info("🔧 Initializing database connection...")
            self.initialize_database()
        
        # Step 1: Deep analysis
        intent_analysis = self.analyze_intent(question)
        search_terms_info = self.extract_search_terms(question)
        search_terms = search_terms_info["keywords"]
        
        logger.info(f"🎯 Intent: {intent_analysis['primary_intent']}")
        logger.info(f"🔍 Search terms: {search_terms}")
        logger.info(f"🏷️ Entities: {search_terms_info.get('entities', {})}")
        
        # Step 2: Smart source prioritization berdasarkan intent
        if intent_analysis["is_aggregation"]:
            # Untuk aggregasi, prioritaskan database
            db_weight = 1.0
            pdf_weight = 0.3
            chat_weight = 0.2
        elif intent_analysis["is_explanation"]:
            # Untuk penjelasan, prioritaskan PDF
            pdf_weight = 1.0
            db_weight = 0.4
            chat_weight = 0.5
        elif intent_analysis["is_data_retrieval"]:
            # Untuk data retrieval, balance semua
            db_weight = 0.7
            pdf_weight = 0.6
            chat_weight = 0.5
        else:
            # Default weights
            db_weight = 0.6
            pdf_weight = 0.6
            chat_weight = 0.5
        public_link_weight = pdf_weight
        external_db_weight = db_weight

        results: Dict[str, Any] = {}

        # Step 3: Parallel search dengan weights.
        # Each source is independent I/O/CPU work, so the sub-searches run
        # concurrently in a small thread pool instead of one after another.
        # hybrid_search already executes off the event-loop thread (called via
        # asyncio.to_thread), so this is a plain ThreadPoolExecutor, not asyncio.
        def _search_pdf() -> tuple:
            logger.info(f"📄 Searching PDF (weight: {pdf_weight})...")
            pdf_docs = self.search_across_collections(
                question,
                collection_ids=collection_ids,
                top_k=int(config.k_per_collection * pdf_weight)
            )
            for doc in pdf_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= pdf_weight
            return 'pdf_documents', pdf_docs

        def _search_db() -> tuple:
            logger.info(f"🗄️ Searching DB (weight: {db_weight})...")
            target_tables = self.get_target_tables(question.lower())
            db_results = self.query_structured_data(search_terms, target_tables)
            for table_name, db_result in db_results.items():
                if hasattr(db_result, 'data'):
                    for record in db_result.data:
                        if 'relevance_score' in record:
                            record['relevance_score'] *= db_weight
            return 'database_results', db_results

        def _search_chat() -> tuple:
            logger.info(f"💬 Searching chat (weight: {chat_weight})...")
            file_filter = self.extract_file_reference(question)
            # Floor of 6: chat chunks are tiny (chat_chunk_size=300), so a
            # weighted k of 1-2 chunks gives the LLM almost no conversation
            # context to work with.
            chat_docs = self.search_across_chat_collections(
                question,
                collection_ids=chat_collection_ids,
                file_filter=file_filter,
                top_k=max(int(config.k_per_collection * chat_weight), 6)
            )
            for doc in chat_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= chat_weight
            return 'chat_documents', chat_docs

        def _search_public_links() -> tuple:
            logger.info(f"🌐 Searching active public links in realtime (weight: {public_link_weight})...")
            # Wider top_k than k_per_collection: public links span many files,
            # and the answer stage can consume up to 8 snippets.
            public_link_docs = self.search_public_links_realtime(
                question,
                public_link_sources=public_link_sources,
                top_k=max(int(config.k_per_collection * public_link_weight), 8),
            )
            for doc in public_link_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= public_link_weight
            return 'public_link_documents', public_link_docs

        def _search_external_db() -> tuple:
            logger.info(f"🗄️ Searching active external database connections in realtime (weight: {external_db_weight})...")
            external_db_docs = self.search_external_db_realtime(
                question,
                db_connections=external_db_connections,
                top_k=max(int(config.k_per_collection * external_db_weight), 8),
            )
            for doc in external_db_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= external_db_weight
            return 'external_db_documents', external_db_docs

        search_tasks = []
        if include_pdf and pdf_weight > 0:
            search_tasks.append(_search_pdf)
        if include_db and db_weight > 0:
            search_tasks.append(_search_db)
        if include_chat and chat_weight > 0:
            search_tasks.append(_search_chat)
        if include_public_links and public_link_weight > 0 and public_link_sources:
            search_tasks.append(_search_public_links)
        if include_external_db and external_db_weight > 0 and external_db_connections:
            search_tasks.append(_search_external_db)

        if search_tasks:
            with ThreadPoolExecutor(max_workers=len(search_tasks)) as executor:
                futures = [executor.submit(task) for task in search_tasks]
                for future in futures:
                    key, value = future.result()
                    results[key] = value

        # Step 4: Cross-source deduplication and ranking
        all_results = self.merge_and_rank_results(results, intent_analysis)

        return {
            **results,
            "search_analysis": {
                "intent": intent_analysis,
                "search_terms": search_terms_info,
                "source_weights": {
                    "pdf": pdf_weight if include_pdf else 0,
                    "db": db_weight if include_db else 0,
                    "chat": chat_weight if include_chat else 0,
                    "public_link": public_link_weight if include_public_links else 0,
                    "external_db": external_db_weight if include_external_db else 0,
                }
            },
            "search_terms": search_terms,
            "target_tables": self.get_target_tables(question.lower()) if include_db else [],
            "merged_results": all_results,
            "has_conflicts": self.detect_conflicts(all_results if isinstance(all_results, list) else [])
        }

    def merge_and_rank_results(self, results: Dict[str, Any], intent_analysis: Dict) -> List[Dict]:
        """Merge results from all sources dengan ranking yang cerdas"""
        merged = []
        
        # Extract all results dengan unified format
        if 'pdf_documents' in results:
            for doc in results['pdf_documents']:
                merged.append({
                    'type': 'pdf',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"PDF: {doc.metadata.get('source', 'Unknown')}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.8  # PDF confidence factor
                })
        
        if 'database_results' in results:
            for table_name, db_result in results['database_results'].items():
                if hasattr(db_result, 'data'):
                    for record in db_result.data:
                        # Create readable content from record
                        content_parts = []
                        for key, value in record.items():
                            if key not in ['search_vector', 'relevance_score']:
                                content_parts.append(f"{key}: {value}")
                        
                        merged.append({
                            'type': 'database',
                            'content': "\n".join(content_parts),
                            'score': record.get('relevance_score', 0),
                            'metadata': {'table': table_name, 'record': record},
                            'source': f"DB: {table_name}",
                            'confidence': record.get('relevance_score', 0) * 0.9  # DB confidence factor
                        })
        
        if 'chat_documents' in results:
            for doc in results['chat_documents']:
                merged.append({
                    'type': 'chat',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"Chat: {doc.metadata.get('source', 'Unknown')}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.7  # Chat confidence factor
                })

        if 'public_link_documents' in results:
            for doc in results['public_link_documents']:
                merged.append({
                    'type': 'public_link',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"Public Link: {doc.metadata.get('public_link_title', doc.metadata.get('source', 'Unknown'))}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.85,
                })

        if 'external_db_documents' in results:
            for doc in results['external_db_documents']:
                merged.append({
                    'type': 'external_db',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"DB: {doc.metadata.get('external_db_label', 'Database')} / {doc.metadata.get('source', 'Unknown')}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.9,  # DB confidence factor
                })

        # Sort by confidence score
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply intent-based boosting
        if intent_analysis['is_aggregation']:
            # Boost database results for aggregation
            for item in merged:
                if item['type'] in ('database', 'external_db'):
                    item['confidence'] *= 1.5
        elif intent_analysis['is_explanation']:
            # Boost PDF results for explanation
            for item in merged:
                if item['type'] == 'pdf':
                    item['confidence'] *= 1.5
        
        # Re-sort after boosting
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        
        return merged[:20]  # Return top 20 results

    def detect_conflicts(self, merged_results: List[Dict[str, Any]]) -> List[Dict]:
        """Enhanced conflict detection - identifies contradictions between sources"""
        conflicts = []
        
        # Extract entities and values from all sources
        entity_values: Dict[str, List[Dict[str, Any]]] = {}  # {entity_key: [{value, source, type, confidence}]}
        
        # 1. Extract numerical values (prices, quantities, counts)
        for result in merged_results:
            content = result.get('content', '')
            source = result.get('source', 'Unknown')
            source_type = result.get('type', 'unknown')
            confidence = result.get('confidence', 0)
            
            # Extract numbers with context (price, quantity, count)
            import re
            
            # Price patterns: "Rp 500000", "$100", "500 ribu"
            price_matches = re.findall(r'(?:Rp\s*|\$)?([0-9.,]+)\s*(?:ribu|juta|rb|jt|k|million)?', content, re.IGNORECASE)
            for match in price_matches:
                key = f"price_{source_type}"
                if key not in entity_values:
                    entity_values[key] = []
                entity_values[key].append({
                    'value': match,
                    'source': source,
                    'type': source_type,
                    'confidence': confidence
                })
            
            # Name conflicts: different person names for same role
            # Pattern: "IT Manager: Ahmad" vs "IT Manager: Budi"
            role_name_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content)
            for role, name in role_name_matches:
                key = f"role_{role.lower().replace(' ', '_')}"
                if key not in entity_values:
                    entity_values[key] = []
                entity_values[key].append({
                    'value': name,
                    'source': source,
                    'type': source_type,
                    'confidence': confidence
                })
        
        # 2. Detect conflicts: same entity with different values
        for entity_key, values in entity_values.items():
            if len(values) > 1:
                unique_values: Dict[str, List[Dict[str, Any]]] = {}
                for item in values:
                    val = item['value']
                    if val not in unique_values:
                        unique_values[val] = []
                    unique_values[val].append(item)
                
                # If multiple distinct values exist for same entity
                if len(unique_values) > 1:
                    conflicts.append({
                        'entity': entity_key,
                        'type': 'value_mismatch',
                        'values': [
                            {
                                'value': k,
                                'sources': [i['source'] for i in v],
                                'source_types': [i['type'] for i in v],
                                'avg_confidence': sum(i['confidence'] for i in v) / len(v)
                            }
                            for k, v in unique_values.items()
                        ],
                        'message': f"Conflicting information for {entity_key}: {list(unique_values.keys())}",
                        'resolution': f"Prioritize {max(unique_values.items(), key=lambda x: sum(i['confidence'] for i in x[1]))[0]} (highest confidence)"
                    })
        
        # 3. Semantic conflicts: contradictory statements
        for i, result1 in enumerate(merged_results[:3]):
            for result2 in merged_results[i+1:4]:
                if result1['type'] != result2['type']:  # Only check across different source types
                    # Check for negation conflicts
                    content1_lower = result1['content'].lower()
                    content2_lower = result2['content'].lower()
                    
                    # Simple contradiction detection
                    if ('tidak' in content1_lower or 'no' in content1_lower) and \
                       ('tidak' not in content2_lower and 'no' not in content2_lower):
                        # Potential contradiction
                        conflicts.append({
                            'type': 'semantic_contradiction',
                            'sources': [result1['source'], result2['source']],
                            'source_types': [result1['type'], result2['type']],
                            'message': f"Potential contradiction between {result1['type']} and {result2['type']}",
                            'resolution': f"Prioritize {result1['type']} (higher confidence: {result1['confidence']:.2%})"
                        })
        
        return conflicts
    
    def rank_and_combine_results(
            self, 
            question: str, 
            pdf_docs: List, 
            db_results: Dict[str, DatabaseResult], 
            chat_docs: List, 
            search_terms: List[str]
        ) -> List[Dict[str, Any]]:
            """
            Gabungkan hasil dari PDF, database, dan chat, lalu beri peringkat berdasarkan relevansi.
            
            Returns:
                List[Dict]: List hasil gabungan yang sudah diurutkan berdasarkan skor relevansi.
            """
            combined = []
            
            # 1. Tambahkan hasil PDF
            for doc in pdf_docs:
                score = doc.metadata.get('similarity_score', 0)
                combined.append({
                    'type': 'pdf',
                    'content': doc.page_content,
                    'score': score,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'Unknown')
                })
            
            # 2. Tambahkan hasil database
            for table_name, db_result in db_results.items():
                for record in db_result.data:
                    # Hitung skor untuk record database berdasarkan kemiripan dengan search_terms
                    record_text = ' '.join([str(v) for v in record.values()])
                    # Simple text match scoring: count matching terms
                    record_lower = record_text.lower()
                    score = sum(1.0 for term in search_terms if term.lower() in record_lower) / max(len(search_terms), 1)
                    combined.append({
                        'type': 'database',
                        'content': record_text,
                        'score': score,
                        'metadata': {'table': table_name, 'record': record},
                        'source': f'Database table: {table_name}'
                    })
            
            # 3. Tambahkan hasil chat
            for doc in chat_docs:
                score = doc.metadata.get('similarity_score', 0)
                combined.append({
                    'type': 'chat',
                    'content': doc.page_content,
                    'score': score,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'Unknown')
                })
            
            # Urutkan berdasarkan skor (descending)
            combined.sort(key=lambda x: x['score'], reverse=True)
            
            return combined
    #     def generate_hybrid_answer(
    #         self, 
    #         hybrid_results: Dict[str, Any], 
    #         question: str,
    #         llm_provider: Optional[str] = None,
    #         llm_model: Optional[str] = None
    #     ) -> Tuple[str, str]:
    #         """
    #         Generate answer combining both structured and unstructured data with relevance scoring.
            
    #         Returns: (answer_text, model_identifier)
    #         """
    #         # Get LLM based on request parameters or defaults
    #         llm, model_id = self.get_llm(llm_provider, llm_model)
            
    #         pdf_docs = hybrid_results['pdf_documents']
    #         db_results = hybrid_results['database_results']
    #         target_tables = hybrid_results.get('target_tables', [])
    #         chat_docs = hybrid_results.get('chat_documents', [])  # NEW

    #         has_pdf_results = len(pdf_docs) > 0
    #         has_db_results = len(db_results) > 0
    #         has_chat_results = len(chat_docs) > 0  # NEW
            
    #         if not has_pdf_results and not has_db_results and not has_chat_results:
    #             return "Maaf, tidak ditemukan informasi yang relevan dalam dokumen, database, maupun chat logs.", model_id

    #         # Prepare context from all sources
    #         context_parts = []

    #         # Add PDF context with confidence scores
    #         if has_pdf_results:
    #             context_parts.append("INFORMASI DARI DOKUMEN:")
    #             for i, doc in enumerate(pdf_docs[:3]): # limit to top 3 PDF results
    #                 source = doc.metadata.get('source', 'Unknown')
    #                 page = doc.metadata.get('page', 'Unknown')
    #                 score = doc.metadata.get('similarity_score', 0)
    #                 truncated_content = self.truncate_context(doc.page_content, max_tokens=150)
    #                 context_parts.append(f"[Dokumen: {source}, Halaman: {page}, Relevansi: {score:.2f}]\n{truncated_content}\n")

    #         # Add DB context with relevance scores
    #         if has_db_results:
    #             context_parts.append("INFORMASI DARI DATABASE:")
                
    #             for table_name, db_result in db_results.items():
    #                 if db_result.record_count > 0:
    #                     # Sort records by relevance_score if available
    #                     sorted_records = sorted(
    #                         db_result.data, 
    #                         key=lambda x: x.get('relevance_score', 0), 
    #                         reverse=True
    #                     )
                        
    #                     context_parts.append(f"\nData dari tabel {table_name}:")
    #                     for i, record in enumerate(sorted_records[:3]): # Limit to top 3 records
    #                         # Filter out internal fields - cleaner format for LLM
    #                         display_fields = {k: v for k, v in record.items() 
    #                                          if not k.startswith('_') and k not in ['search_vector', 'relevance_score', 'created_at']}
    #                         record_str = ", ".join(f"{k}: {v}" for k, v in list(display_fields.items())[:6])
    #                         context_parts.append(f"• {record_str}")

    #                     if db_result.record_count > 3:
    #                         context_parts.append(f"(dan {db_result.record_count - 3} record lainnya)")

    
    #         if has_chat_results:
    #             context_parts.append("\nINFORMASI DARI CHAT LOGS:")
                
    #             # Prioritize chunks that contain query keywords in content
    #             question_lower = question.lower()
    #             query_words = [w for w in question_lower.split() if len(w) > 2]
                
    #             # Re-sort chat docs by keyword relevance to answer the specific question
    #             def keyword_priority(doc):
    #                 content = doc.page_content.lower()
    #                 matches = sum(1 for word in query_words if word in content)
    #                 return (matches, doc.metadata.get('similarity_score', 0))
                
    #             sorted_chat_docs = sorted(chat_docs, key=keyword_priority, reverse=True)
                
    #             # Take top 5 chat results for more context
    #             for i, doc in enumerate(sorted_chat_docs[:5]):
    #                 source = doc.metadata.get('source', 'Unknown')
    #                 platform = doc.metadata.get('platform', 'unknown')
    #                 participants = doc.metadata.get('participants', '')
    #                 score = doc.metadata.get('similarity_score', 0)
    #                 time_start = doc.metadata.get('time_range_start', '')
                    
    #                 # Increase token limit for chat to capture more context
    #                 truncated_content = self.truncate_context(doc.page_content, max_tokens=400)
    #                 context_parts.append(f"[Sumber: {source}]")
    #                 context_parts.append(f"{truncated_content}\n")

    #         context = "\n".join(context_parts)
    #         context = self.truncate_context(context, max_tokens=900)  # Increased for better chat context

    #         # Extract main keyword from question for focused answering
    #         question_lower = question.lower()
    #         main_keyword = ""
    #         for keyword in ['buyback cash', 'buyback debt switch', 'lelang', 'auction', 'settlement', 'lpdu', 'sbn', 'sun']:
    #             if keyword in question_lower:
    #                 main_keyword = keyword
    #                 break
            
    #         # Improved prompt that focuses on the specific question
    #         if main_keyword:
    #             prompt_template = """Berikan definisi atau penjelasan tentang "{keyword}" berdasarkan konteks berikut.

    # Konteks:
    # {context}

    # Pertanyaan: {question}

    # Jawaban tentang {keyword}:"""
    #             prompt = prompt_template.format(keyword=main_keyword, context=context, question=question)
    #         else:
    #             # Check if this is primarily a chat question
    #             if has_chat_results and not has_pdf_results and not has_db_results:
    #                 prompt_template = """Ekstrak informasi yang diminta dari percakapan chat berikut. Berikan jawaban yang spesifik dan langsung.

    # Percakapan:
    # {context}

    # Pertanyaan: {question}

    # Jawaban (langsung dan spesifik):"""
    #             # Check if this is primarily a database question
    #             elif has_db_results and not has_pdf_results:
    #                 prompt_template = """Berdasarkan data berikut, jawab pertanyaan dengan format yang jelas dan informatif.

    # Data:
    # {context}

    # Pertanyaan: {question}

    # Jawaban:"""
    #             else:
    #                 prompt_template = """Jawab pertanyaan berikut berdasarkan konteks. Jawab dalam Bahasa Indonesia.

    # Konteks:
    # {context}

    # Pertanyaan: {question}

    # Jawaban:"""
    #             prompt = prompt_template.format(context=context, question=question)
            
    #         logger.debug(f"Generated prompt length: {len(prompt)} chars, using model: {model_id}")

    #         try:
    #             result = llm.invoke(prompt)
                
    #             # Handle different response types (ChatOllama returns AIMessage, HuggingFace returns str)
    #             if hasattr(result, 'content'):
    #                 answer = result.content.strip()
    #             else:
    #                 answer = str(result).strip()
                
    #             # Validate answer - if it looks garbled, return fallback
    #             if self._is_garbled_output(answer):
    #                 logger.warning(f"Garbled output detected: {answer[:100]}")
    #                 fallback = self._generate_fallback_answer(hybrid_results, question)
    #                 return fallback, model_id
                
    #             return answer, model_id
    #         except Exception as e:
    #             logger.error(f"LLM generation failed: {str(e)}")
    #             fallback = self._generate_fallback_answer(hybrid_results, question)
    #             return fallback, model_id

#     def generate_hybrid_answer(
#         self, 
#         hybrid_results: Dict[str, Any], 
#         question: str,
#         llm_provider: Optional[str] = None,
#         llm_model: Optional[str] = None
#     ) -> Tuple[str, str]:
#         """
#         Generate answer combining both structured and unstructured data with relevance scoring.
        
#         Returns: (answer_text, model_identifier)
#         """
#         # Get LLM based on request parameters or defaults
#         llm, model_id = self.get_llm(llm_provider, llm_model)
        
#         pdf_docs = hybrid_results['pdf_documents']
#         db_results = hybrid_results['database_results']
#         target_tables = hybrid_results.get('target_tables', [])
#         chat_docs = hybrid_results.get('chat_documents', [])  # NEW
#         combined_results = hybrid_results.get('combined_results', [])  # Gabungan hasil
        
#         has_pdf_results = len(pdf_docs) > 0
#         has_db_results = len(db_results) > 0
#         has_chat_results = len(chat_docs) > 0  # NEW
        
#         if not has_pdf_results and not has_db_results and not has_chat_results:
#             return "Maaf, tidak ditemukan informasi yang relevan dalam dokumen, database, maupun chat logs.", model_id

#         # Siapkan konteks dari semua sumber yang sudah digabung dan diurutkan
#         context_parts = []
#         for i, result in enumerate(combined_results[:10]):  # Ambil 10 teratas
#             source_type = result['type']
#             source = result['source']
#             content = result['content']
#             score = result['score']
            
#             context_parts.append(f"[Sumber {i+1}: {source} ({source_type}), Skor Relevansi: {score:.3f}]\n{content}\n")
        
#         context = "\n".join(context_parts)
#         context = self.truncate_context(context, max_tokens=900)  # Increased for better chat context

#         # Prompt yang lebih baik untuk menggabungkan informasi
#         prompt_template = """Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan informasi dari berbagai sumber (PDF, database, chat). 
# Informasi dari sumber-sumber tersebut diberikan di bawah ini, sudah diurutkan berdasarkan relevansi.

# Konteks:
# {context}

# Pertanyaan: {question}

# Instruksi:
# 1. Jawablah pertanyaan dengan jelas dan singkat dalam Bahasa Indonesia.
# 2. Jika informasi dari beberapa sumber bertentangan, utamakan sumber dengan skor relevansi tertinggi.
# 3. Jika tidak ada informasi yang cukup, katakan bahwa Anda tidak tahu.
# 4. Sebutkan sumber informasi yang Anda gunakan dalam jawaban (misalnya: menurut dokumen PDF, dari database, atau dari chat).

# Jawaban:"""
        
#         prompt = prompt_template.format(context=context, question=question)
        
#         logger.debug(f"Generated prompt length: {len(prompt)} chars, using model: {model_id}")

#         try:
#             result = llm.invoke(prompt)
            
#             # Handle different response types (ChatOllama returns AIMessage, HuggingFace returns str)
#             if hasattr(result, 'content'):
#                 answer = result.content.strip()
#             else:
#                 answer = str(result).strip()
            
#             # Validasi jawaban
#             if self._is_garbled_output(answer):
#                 logger.warning(f"Garbled output detected: {answer[:100]}")
#                 fallback = self._generate_fallback_answer(hybrid_results, question)
#                 return fallback, model_id
            
#             return answer, model_id
#         except Exception as e:
#             logger.error(f"LLM generation failed: {str(e)}")
#             fallback = self._generate_fallback_answer(hybrid_results, question)
#             return fallback, model_id   
    
    # processor.py - perbaiki generate_hybrid_answer
    def generate_hybrid_answer(
        self, 
        hybrid_results: Dict[str, Any], 
        question: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Enhanced hybrid answer generation dengan conflict resolution and comprehensive metadata"""
        
        llm, model_id = self.get_llm(llm_provider, llm_model)
        
        intent_analysis = hybrid_results.get('search_analysis', {}).get('intent', {})
        merged_results = hybrid_results.get('merged_results', [])
        
        # Enhanced conflict detection
        conflicts = self.detect_conflicts(merged_results)
        
        # Track processing steps for observability
        processing_steps = ["Intent analysis completed", "Sources merged and ranked"]
        
        # Check if we have chat or DB results with good info - prioritize them over PDF
        question_lower = question.lower()
        # NOTE: "yang" was removed from this list — it's one of the most
        # common words in Indonesian (relative pronoun "that/which"), so it
        # matched most questions and mis-triggered person-query handling below.
        is_person_query = any(kw in question_lower for kw in ['siapa', 'who', 'handle', 'contact'])
        
        # Track exact matches and boost application
        exact_matches = []
        boost_applied = {"exact_match": 0.0, "person_query": 0.0, "brand_whitelist": 0.0}
        
        # Detect exact matches in results
        search_terms = hybrid_results.get('search_terms', [])
        for result in merged_results[:5]:
            content_lower = result.get('content', '').lower()
            for term in search_terms:
                if term.lower() in content_lower:
                    exact_matches.append({"term": term, "source": result.get('source', 'Unknown'), "type": result['type']})
                    # Check if boost was applied (from database exact match algorithm)
                    if result['type'] == 'database' and result.get('confidence', 0) > 0.9:
                        boost_applied["exact_match"] += 50.0  # +50 boost from database.py
        
        processing_steps.append(f"Detected {len(exact_matches)} exact matches")
        
        # For person queries, prioritize chat and DB results
        if is_person_query and merged_results:
            # Reorder to prioritize chat/db, but keep every other type
            # (pdf, public_link, external_db, ...) instead of dropping them —
            # a plain equality filter here silently discarded any type it
            # didn't know about.
            chat_results = [r for r in merged_results if r['type'] == 'chat']
            db_results = [r for r in merged_results if r['type'] in ('database', 'external_db')]
            other_results = [r for r in merged_results if r['type'] not in ('chat', 'database', 'external_db')]
            merged_results = chat_results + db_results + other_results
            boost_applied["person_query"] = 1.8  # Person query boost multiplier
            processing_steps.append("Person query detected - prioritized chat/DB results")
        
        # Check if confidence is too low and no chat/db results - use direct extraction
        # Use a dynamic threshold: lower for smarter models like Gemini to leverage generative reasoning
        threshold = 0.12 if 'gemini' in model_id.lower() else 0.18
        if merged_results and merged_results[0]['confidence'] < threshold:
            if merged_results[0]['type'] == 'pdf' and not any(r['type'] in ['chat', 'database'] for r in merged_results[:3]):
                logger.warning(f"Low confidence ({merged_results[0]['confidence']:.2%}), using direct extraction")
                return self._extract_direct_answer(merged_results[0], question), model_id, {
                    "intent": "direct_extraction",
                    "sources_used": {"pdf": 1, "database": 0, "chat": 0},
                    "top_confidence": merged_results[0]['confidence'],
                    "conflicts_detected": False,
                    "extraction_method": "direct"
                }
        
        # Prepare context dengan prioritization - SHORTER for small models
        context_parts = []
        source_breakdown = {"pdf": 0, "database": 0, "chat": 0, "public_link": 0, "external_db": 0}
        
        # Limit to top 3 results and shorter snippets for small models.
        # Large-context models (Gemini) get full chunks: truncating to a few
        # hundred chars discards the very passage retrieval worked to find.
        is_small_model = 'flan-t5' in model_id.lower()
        max_results = 3 if is_small_model else 8
        base_max_content_len = 150 if is_small_model else 2000
        # Tabular sources need a bigger allowance than prose: a table chunk
        # (now sized to hold a whole small table, see external_db_chunk_size)
        # must not be re-truncated here, or the same "only some rows survive"
        # problem the bigger chunk size was meant to fix just reappears.
        structured_max_content_len = 150 if is_small_model else 6000

        for i, result in enumerate(merged_results[:max_results]):
            source_type = result['type']
            source_breakdown[source_type] += 1

            if is_small_model:
                # Small-context models need keyword-window extraction
                content_snippet = self._extract_relevant_snippet(result['content'], question)
            else:
                # Large-context models (Gemini) get the retrieved chunk as-is,
                # matching the research notebook pipeline: keyword-window
                # extraction on messy PDF text often clips out the relevant
                # passage that retrieval already found.
                content_snippet = result['content']

            max_content_len = (
                structured_max_content_len
                if source_type in ('database', 'external_db')
                else base_max_content_len
            )
            # Truncate if too long based on model
            if len(content_snippet) > max_content_len:
                content_snippet = content_snippet[:max_content_len] + "..."

            context_parts.append(
                f"[Sumber {i+1} — {result['source']}]:\n{content_snippet}"
            )

        context = "\n\n---\n\n".join(context_parts)
        
        # Build enhanced prompt berdasarkan intent
        # Aggregation prompt demands numeric computation and treats "database"
        # as the source of truth — that's only correct when the request
        # explicitly scoped the search to database alone (source of truth =
        # what the user selected). If nothing/multiple sources were selected,
        # or the selection wasn't database-only, fall through to the general
        # prompt so words like "jumlah desimal" in a document/chat question
        # don't force a numeric-only answer style.
        source_weights = hybrid_results.get('search_analysis', {}).get('source_weights', {})
        selected_sources = {k for k, w in source_weights.items() if w and w > 0}
        db_is_sole_source = selected_sources in ({'db'}, {'external_db'})
        has_db_results = source_breakdown.get('database', 0) + source_breakdown.get('external_db', 0) > 0

        # Sole-source-type detection: when the user filtered the request down
        # to exactly one source type, the prompt wording should match that
        # source's shape ("percakapan chat" vs "dokumen" vs "data tabel")
        # instead of always saying "dokumen". None covers multi-source /
        # unfiltered requests, which must keep the original combined wording
        # unchanged — see _source_prompt_vocab.
        sole_source_type = None
        if selected_sources == {'pdf'}:
            sole_source_type = 'pdf'
        elif selected_sources == {'chat'}:
            sole_source_type = 'chat'
        elif selected_sources == {'public_link'}:
            sole_source_type = 'public_link'
        elif db_is_sole_source:
            sole_source_type = 'database'

        if intent_analysis.get('is_aggregation') and db_is_sole_source and has_db_results:
            prompt = self._build_aggregation_prompt(context, question, conflicts)
        elif intent_analysis.get('is_comparison'):
            prompt = self._build_comparison_prompt(context, question, conflicts)
        elif intent_analysis.get('is_explanation'):
            prompt = self._build_explanation_prompt(context, question, conflicts, sole_source_type=sole_source_type)
        else:
            prompt = self._build_general_prompt(context, question, conflicts, source_breakdown, sole_source_type=sole_source_type)
        
        try:
            result = llm.invoke(prompt)
            answer = result.content.strip() if hasattr(result, 'content') else str(result).strip()
            
            # Validasi dan post-processing
            answer = self._validate_and_clean_answer(answer, question, merged_results)
            processing_steps.append("Answer validated and cleaned")
            
            # Generate comprehensive metadata untuk response
            answer_metadata = {
                "confidence_score": merged_results[0]['confidence'] if merged_results else 0,
                "primary_intent": intent_analysis.get('primary_intent', 'data_retrieval'),
                "sources_used": source_breakdown,
                "search_strategy": "hybrid_multi_source",
                "conflicts_detected": len(conflicts) > 0,
                "exact_matches": exact_matches[:10],  # Top 10 exact matches
                "boost_applied": boost_applied,
                "ranking_algorithm": "weighted_scoring_with_intent",
                "processing_steps": processing_steps,
                "total_results_processed": len(merged_results),
                "conflict_details": conflicts if conflicts else [],
                "model_used": model_id,
                "is_person_query": is_person_query
            }
            
            return answer, model_id, answer_metadata
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            processing_steps.append(f"LLM generation failed: {str(e)[:100]}")
            return self._generate_fallback_answer(hybrid_results, question), model_id, {
                "error": str(e),
                "processing_steps": processing_steps,
                "sources_used": {"pdf": 0, "database": 0, "chat": 0},
                "confidence_score": 0,
                "conflicts_detected": False
            }

    def _source_prompt_vocab(self, sole_source_type: Optional[str]) -> Dict[str, str]:
        """Wording per jenis source, dipakai hanya kalau user memfilter request
        ke tepat satu source type (sole_source_type). None (multi-source atau
        tidak difilter) selalu jatuh ke wording 'pdf' — perilaku lama, tidak
        berubah — supaya jalur gabungan tetap identik dengan sebelumnya."""
        vocab = {
            'pdf': {
                'noun': 'dokumen',
                'label': 'DOKUMEN',
                'not_found': 'Informasi ini tidak ditemukan dalam dokumen yang diunggah.',
                'extra_instruction': '',
            },
            'public_link': {
                'noun': 'dokumen dari tautan publik',
                'label': 'DOKUMEN DARI TAUTAN PUBLIK',
                'not_found': 'Informasi ini tidak ditemukan dalam dokumen tautan publik yang tersedia.',
                'extra_instruction': '',
            },
            'chat': {
                'noun': 'percakapan chat',
                'label': 'PERCAKAPAN CHAT',
                'not_found': 'Informasi ini tidak ditemukan dalam percakapan chat yang tersedia.',
                'extra_instruction': '- Percakapan chat bisa informal/singkatan/emoji — tetap ekstrak maknanya, jangan tuntut struktur formal.',
            },
            'database': {
                'noun': 'data tabel',
                'label': 'DATA TABEL',
                'not_found': 'Informasi ini tidak ditemukan dalam data yang tersedia.',
                'extra_instruction': '- Jika data berupa angka/tabel, tampilkan persis apa adanya tanpa dibulatkan sendiri.',
            },
        }
        return vocab.get(sole_source_type, vocab['pdf'])

    def _build_aggregation_prompt(self, context: str, question: str, conflicts: List) -> str:
        """Build prompt untuk aggregation queries"""
        conflict_note = ""
        if conflicts:
            conflict_note = "\nPERHATIAN: Ditemukan perbedaan data antara sumber. Gunakan sumber database sebagai referensi utama."
        
        return f"""Anda adalah asisten analisis data. Hitung dan berikan jawaban numerik berdasarkan data berikut.

    DATA YANG TERSEDIA (sudah diurutkan berdasarkan kepercayaan):
    {context}

    PERTANYAAN: {question}
    {conflict_note}

    PETUNJUK:
    1. Hitung nilai yang diminta (total, rata-rata, jumlah, dll)
    2. Gunakan format: "Berdasarkan data dari [sumber], [jawaban numerik]"
    3. Jika ada perbedaan data, sebutkan perbedaan tersebut
    4. Sertakan satuan jika relevan (Rp, unit, orang, dll)
    5. Berikan jawaban dalam Bahasa Indonesia
    {OFF_TOPIC_INSTRUCTION_ID}

    JAWABAN:"""

    def _build_general_prompt(
        self,
        context: str,
        question: str,
        conflicts: List,
        source_breakdown: Dict,
        sole_source_type: Optional[str] = None,
    ) -> str:
        """Build prompt untuk general queries"""
        vocab = self._source_prompt_vocab(sole_source_type)
        extra_instruction = f"\n{vocab['extra_instruction']}" if vocab['extra_instruction'] else ""

        return f"""Jawab pertanyaan berdasarkan {vocab['noun']} berikut SAJA.

{vocab['label']}:
{context}

Pertanyaan: {question}

INSTRUKSI PENTING:
- Jawab HANYA berdasarkan isi {vocab['noun']} di atas
- Jika {vocab['noun']} memuat informasi yang berkaitan sebagian dengan pertanyaan, jawab sebaik mungkin dari bagian yang relevan itu dan sebutkan bahwa informasinya terbatas
- Jika {vocab['noun']} mengandung angka, tabel, atau data — ekstrak dan tampilkan langsung dalam jawaban
- Katakan "{vocab['not_found']}" HANYA jika tidak ada satu pun bagian {vocab['noun']} yang berkaitan dengan pertanyaan
- JANGAN gunakan pengetahuan umum atau informasi dari luar {vocab['noun']}{extra_instruction}
{OFF_TOPIC_INSTRUCTION_ID}
- Jawab ringkas dan jelas dalam Bahasa Indonesia

Jawaban:"""

    def _build_comparison_prompt(self, context: str, question: str, conflicts: List) -> str:
        """Build prompt untuk comparison queries"""
        return f"""Anda adalah asisten analisis yang membantu membandingkan data.

DATA YANG TERSEDIA:
{context}

PERTANYAAN: {question}

INSTRUKSI:
1. Bandingkan data yang diminta
2. Highlight perbedaan utama
3. Gunakan format tabel atau bullet points jika sesuai
4. Berikan kesimpulan singkat
5. Jawab dalam Bahasa Indonesia
{OFF_TOPIC_INSTRUCTION_ID}

JAWABAN:"""

    def _build_explanation_prompt(
        self,
        context: str,
        question: str,
        conflicts: List,
        sole_source_type: Optional[str] = None,
    ) -> str:
        """Build prompt untuk explanation queries"""
        vocab = self._source_prompt_vocab(sole_source_type)
        extra_instruction = f"\n{vocab['extra_instruction']}" if vocab['extra_instruction'] else ""

        return f"""Anda adalah asisten yang menjawab HANYA berdasarkan {vocab['noun']} yang diberikan.

{vocab['label']} YANG TERSEDIA:
{context}

PERTANYAAN: {question}

INSTRUKSI PENTING:
- Jawab HANYA berdasarkan informasi di {vocab['noun']} yang sudah tersedia.
- Jika {vocab['noun']} memuat informasi yang berkaitan sebagian dengan pertanyaan, jawab sebaik mungkin dari bagian yang relevan itu dan sebutkan bahwa informasinya terbatas.
- Katakan "{vocab['not_found']}" HANYA jika tidak ada satu pun bagian {vocab['noun']} yang berkaitan dengan pertanyaan.
- JANGAN gunakan pengetahuan umum atau informasi dari luar {vocab['noun']}
- Gunakan bullet points atau daftar poin-poin HANYA jika sumber asli menyebutkan daftar terpisah atau beberapa item yang terdaftar secara terpisah. Jika sumber asli berisi narasi atau teks aslinya berupa penjelasan paragraf mengalir terus-menerus (seperti paragraf utuh tunggal), pertahankan sebagai paragraf mengalir/narasi utuh apa adanya tanpa dibuat menjadi poin-poin terpisah.{extra_instruction}
{OFF_TOPIC_INSTRUCTION_ID}
- Jawab dengan jelas dalam Bahasa Indonesia

JAWABAN:"""

    # ===================== SKILL: Reference Framework Gap Analysis =====================
    # Generic "Skill" plumbing: skill_id picks a prompt builder + output schema.
    # "compliance_gap_check" (Skill 1) is validated first via ISO 27001/9001 —
    # nothing below is ISO-specific, ISO is just the first framework_name used.
    # "scenario_regulatory_impact" (Skill 2) is scaffolded elsewhere and not
    # wired into this orchestration yet — it is not validated for real use.

    GAP_CHECK_BATCH_SIZE = 8
    GAP_CHECK_MAX_WORKERS = 3  # cap concurrency to respect Gemini rate limits
    # Batches now multiply by len(target_collection_ids) (one guideline vs N
    # files), so a flat cap of 3 would serialize multi-target runs more and
    # more the more files are compared. Scale with target count, but stay
    # under a hard ceiling so we don't blow past Gemini's rate limits either.
    GAP_CHECK_MAX_WORKERS_CEILING = 6
    # Chunking is 500 chars/chunk (config.py), so the old 12,000-char cap kept
    # only ~24 chunks — often just a framework doc's title/scope/definitions,
    # cut off before the actual itemized clause/control list. Gemini's context
    # window comfortably fits far more, so extraction gets the real content.
    GAP_CHECK_REFERENCE_MAX_CHARS = 60000

    def _get_reference_text(self, reference_collection_ids: List[str], max_chars: Optional[int] = None) -> str:
        """Concatenate the raw text of one or more reference collections."""
        if max_chars is None:
            max_chars = self.GAP_CHECK_REFERENCE_MAX_CHARS
        parts = []
        for cid in reference_collection_ids:
            vector_store = self.get_vector_store(cid)
            if not vector_store:
                logger.warning(f"_get_reference_text: no vector store found for reference collection {cid}")
                continue
            try:
                docs = list(vector_store.docstore._dict.values())
            except Exception as e:
                logger.warning(f"_get_reference_text: could not read docstore for {cid}: {e}")
                docs = []
            if not docs:
                logger.warning(f"_get_reference_text: reference collection {cid} has no indexed documents")
            for doc in docs:
                parts.append(doc.page_content)
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            logger.info(f"_get_reference_text: truncating reference text from {len(text)} to {max_chars} chars")
            text = text[:max_chars] + "..."
        return text

    def _parse_json_list(self, raw: str) -> List[str]:
        """Best-effort JSON array parsing — strips markdown fences the model
        might add despite instructions, falls back to substring extraction.
        Also unwraps a single-key object (e.g. {"items": [...]})  the model
        may return instead of a bare array despite being asked for one."""
        text = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                list_values = [v for v in data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    return [str(x) for x in list_values[0]]
        except Exception:
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except Exception:
                pass
        return []

    def _parse_json_objects(self, raw: str) -> List[Dict[str, Any]]:
        """Same as _parse_json_list but for an array of objects."""
        text = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def extract_framework_items(self, reference_collection_ids: List[str], framework_name: str) -> List[str]:
        """Ask the LLM to enumerate discrete requirement/control items from the
        reference collection(s). Generic — works for any standard/framework
        the user uploads, not just ISO."""
        reference_text = self._get_reference_text(reference_collection_ids)
        if not reference_text.strip():
            return []

        prompt = f"""Anda membaca dokumen standar/framework bernama "{framework_name}".

DOKUMEN:
{reference_text}

TUGAS: Daftar SEMUA item/klausul/kontrol/requirement yang disebutkan sebagai satuan terpisah di dokumen ini.

FORMAT WAJIB: kembalikan HANYA JSON array of string, tanpa markdown fence, tanpa penjelasan tambahan.
Contoh: ["A.5.1 Kebijakan keamanan informasi", "A.5.2 Peran dan tanggung jawab keamanan informasi"]

JSON:"""

        llm, _ = self.get_llm(provider="gemini")
        raw = ""
        try:
            result = llm.invoke(prompt)
            raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
            items = self._parse_json_list(raw)
        except Exception as e:
            logger.error(f"extract_framework_items failed: {e}")
            items = []

        # Retry once on empty/malformed output — mirrors the retry already
        # done per-batch below, so a single bad generation doesn't sink the
        # whole run (this step has no batching to fall back on otherwise).
        if not items:
            logger.warning(
                f"extract_framework_items: first attempt returned no items "
                f"(raw response length={len(raw)}); retrying once"
            )
            try:
                retry_prompt = prompt + "\n\nPERHATIAN: balas HANYA dengan JSON array of string yang valid, tanpa teks lain, tanpa markdown fence."
                result = llm.invoke(retry_prompt)
                raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                items = self._parse_json_list(raw)
            except Exception as e:
                logger.error(f"extract_framework_items retry failed: {e}")
                items = []
            if not items:
                logger.error(
                    f"extract_framework_items: retry also returned no items; "
                    f"raw response (first 500 chars): {raw[:500]!r}"
                )
        return items

    def _build_gap_check_batch_prompt(self, framework_name: str, batch_items: List[Dict[str, str]]) -> str:
        """batch_items: list of {"label": ..., "target_context": ...}. Generic
        prompt — takes framework_name/context as parameters, no ISO-specific text."""
        items_block = "\n\n".join(
            f"ITEM {i+1}: {it['label']}\nBUKTI DARI DOKUMEN PERUSAHAAN:\n"
            f"{it['target_context'] or '(tidak ditemukan bukti terkait)'}"
            for i, it in enumerate(batch_items)
        )
        return f"""Anda adalah asisten audit compliance. Standar/framework yang dipakai: "{framework_name}".
Sumber standar ini kemungkinan berupa ringkasan/interpretasi pihak ketiga, BUKAN teks resmi berlisensi —
jangan berpura-pura mengutip klausul resmi kata demi kata.

Untuk setiap ITEM di bawah, tentukan apakah BUKTI dari dokumen perusahaan menunjukkan item itu sudah terpenuhi.

{items_block}

Status yang valid:
- "met": bukti jelas menunjukkan item terpenuhi
- "partial": ada bukti tapi tidak lengkap
- "not_met": tidak ada bukti relevan
- "unknown": tidak bisa disimpulkan dari bukti yang ada

FORMAT WAJIB: HANYA JSON array, tanpa markdown fence, satu object per item, urut sesuai urutan ITEM di atas:
[{{"label": "...", "status": "met|partial|not_met|unknown", "evidence": "kutipan singkat dari BUKTI atau kosong", "recommendation": "rekomendasi singkat kalau belum met, atau kosong"}}]

JSON:"""

    def run_compliance_gap_check(
        self,
        reference_collection_ids: List[str],
        target_collection_ids: List[str],
        framework_name: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Skill 1 — Compliance Gap Check. Generic map-per-batch orchestration
        (not an autonomous agent): extract reference items ONCE, then check
        each target collection independently against that same item list —
        so one guideline can be compared against several company files and
        each item comes back tagged with which file/collection it was
        checked against, instead of one merged verdict across all of them.
        Batches (item groups x target collections) run concurrently.
        Returns (items, disclaimer)."""
        items_labels = self.extract_framework_items(reference_collection_ids, framework_name)
        disclaimer = (
            f"Analisis ini berdasarkan dokumen \"{framework_name}\" yang diupload sebagai referensi "
            "— kemungkinan ringkasan/interpretasi pihak ketiga, bukan teks standar resmi berlisensi. "
            "Hasil ini bersifat bantuan awal, bukan audit/sertifikasi resmi."
        )
        if not items_labels:
            return [], disclaimer

        batches = [
            items_labels[i:i + self.GAP_CHECK_BATCH_SIZE]
            for i in range(0, len(items_labels), self.GAP_CHECK_BATCH_SIZE)
        ]

        def process_batch(batch_labels: List[str], target_collection_id: str) -> List[Dict[str, Any]]:
            batch_items = []
            for label in batch_labels:
                try:
                    results = self.search_across_collections(
                        label, collection_ids=[target_collection_id], top_k=3
                    )
                except Exception as e:
                    logger.warning(f"gap-check retrieval failed for item '{label}' in {target_collection_id}: {e}")
                    results = []
                target_context = "\n---\n".join(
                    doc.page_content[:1500] for doc in results[:3]
                )
                # Per-chunk "source" filename is already set at ingestion time
                # (utils.py) and preserved through retrieval — surface it here
                # so evidence points at an actual file, not just the collection id.
                source_files = []
                for doc in results[:3]:
                    src = doc.metadata.get("source")
                    if src and src not in source_files:
                        source_files.append(src)
                batch_items.append({
                    "label": label,
                    "target_context": target_context,
                    "source_files": source_files,
                })

            prompt = self._build_gap_check_batch_prompt(framework_name, batch_items)
            llm, _ = self.get_llm(provider="gemini")
            parsed: List[Dict[str, Any]] = []
            try:
                result = llm.invoke(prompt)
                raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                parsed = self._parse_json_objects(raw)
            except Exception as e:
                logger.error(f"gap-check batch LLM call failed: {e}")

            # Retry the batch once on parse failure / count mismatch — retry
            # per-batch, not per-run, so one bad batch doesn't cost a full re-run.
            if len(parsed) != len(batch_labels):
                try:
                    retry_prompt = prompt + "\n\nPERHATIAN: balas HANYA dengan JSON array yang valid, tanpa teks lain."
                    result = llm.invoke(retry_prompt)
                    raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                    parsed = self._parse_json_objects(raw)
                except Exception as e:
                    logger.error(f"gap-check batch retry failed: {e}")

            out = []
            for i, label in enumerate(batch_labels):
                match = parsed[i] if i < len(parsed) else {}
                status = str(match.get("status", "unknown")).lower()
                if status not in ("met", "partial", "not_met", "unknown"):
                    status = "unknown"
                source_files = batch_items[i]["source_files"]
                citation = ", ".join(source_files) if source_files else f"target_collection:{target_collection_id}"
                out.append({
                    "label": match.get("label") or label,
                    "status": status,
                    "evidence": match.get("evidence") or None,
                    "source_citation": citation,
                    "recommendation": match.get("recommendation") or None,
                    "target_collection_id": target_collection_id,
                })
            return out

        all_items: List[Dict[str, Any]] = []
        effective_workers = max(1, min(
            self.GAP_CHECK_MAX_WORKERS_CEILING,
            self.GAP_CHECK_MAX_WORKERS * len(target_collection_ids),
        ))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(process_batch, batch, target_collection_id)
                for target_collection_id in target_collection_ids
                for batch in batches
            ]
            for future in futures:
                try:
                    all_items.extend(future.result())
                except Exception as e:
                    logger.error(f"gap-check batch future failed: {e}")

        return all_items, disclaimer

    def _validate_and_clean_answer(self, answer: str, question: str, results: List[Dict]) -> str:
        """Validate dan clean LLM answer"""
        
        # Check for common LLM failure patterns including prompt echo from small models
        failure_patterns = [
            "maaf,", "tidak tahu", "tidak menemukan", "no information",
            "based on the context", "context:", "question:", "answer:",
            "- hanya berdasar", "jangan gunakan", "instruksi penting",
            "jawab hanya berdasarkan", "informasi tidak ditemukan dalam dokumen yang diunggah"
        ]
        
        if any(pattern in answer.lower() for pattern in failure_patterns):
            # Generate answer from best result
            if results:
                best_result = results[0]
                return self._format_result_as_answer(best_result, question)
        
        # Ensure answer is not too short
        if len(answer.split()) < 5:
            if results:
                best_result = results[0]
                return f"Informasi dari {best_result['source']}:\n{best_result['content']}"
        
        return answer

    def _format_result_as_answer(self, result: Dict, question: str) -> str:
        """Format single result as answer using context-aware snippet extraction"""
        source = result['source']
        confidence = result['confidence']
        
        if result['type'] == 'database':
            # Format database record
            return f"Berdasarkan data dari {source} (akurasi: {confidence:.0%}):\n\n{result['content']}"
        elif result['type'] == 'pdf':
            snippet = self._extract_relevant_snippet(result['content'], question)
            if not snippet:
                snippet = result['content']
            return f"Informasi dari dokumen {source} (relevansi: {confidence:.0%}):\n\n{snippet}"
        else:
            return f"Dari {source}:\n\n{result['content']}"
    
    def _extract_relevant_snippet(self, content: str, question: str) -> str:
        """Extract most relevant part of content based on question keywords"""
        # Get question keywords
        question_lower = question.lower()
        keywords = [w for w in question_lower.split() if len(w) > 2]
        
        # Add special keywords for person queries
        if any(kw in question_lower for kw in ['siapa', 'who', 'handle']):
            # Extract names (capitalized words, likely person names)
            import re
            name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            names = re.findall(name_pattern, content)
            if names:
                # Return sentences containing names
                for name in names[:3]:  # Top 3 names
                    for sent in content.split('.'):
                        if name in sent:
                            return sent.strip()[:300]
        
        # Default max length for snippets (increased to prevent truncation)
        max_len = 3000
        
        # For chat content, extract lines with keywords
        if '[' in content and ']' in content:  # Chat format detection
            lines = content.split('\n')
            best_lines = []
            for line in lines:
                line_lower = line.lower()
                score = sum(1 for kw in keywords if kw in line_lower)
                if score > 0:
                    best_lines.append((score, line))
            
            if best_lines:
                best_lines.sort(reverse=True, key=lambda x: x[0])
                return '\n'.join([line for _, line in best_lines[:15]])
        
        # Split content into sentences
        sentences = content.split('.')
        
        # Score each sentence based on keyword matches
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        # If found relevant sentence, return it with context
        if best_sentence and best_score > 0:
            # Find position in original text
            start_idx = content.find(best_sentence)
            if start_idx >= 0:
                # Get some context before and after
                context_start = max(0, start_idx - 300)
                context_end = min(len(content), start_idx + len(best_sentence) + 1500)
                snippet = content[context_start:context_end].strip()
                if len(snippet) > max_len:
                    snippet = snippet[:max_len] + "..."
                return snippet
        
        # Fallback to first N characters
        return content[:max_len] + ("..." if len(content) > max_len else "")
    
    def _extract_direct_answer(self, result: Dict, question: str) -> str:
        """Extract direct answer from result for factoid questions"""
        content = result['content']
        source = result['source']
        
        # Try to find phone numbers
        phone_pattern = r'(\+?62[-\s]?\d{3}[-\s]?\d{4}[-\s]?\d{4}|\d{4}|ext\.\s*\d+)'
        phones = re.findall(phone_pattern, content, re.IGNORECASE)
        
        # Try to find emails
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        emails = re.findall(email_pattern, content)
        
        # Detect question type
        question_lower = question.lower()
        
        if 'nomor' in question_lower or 'telepon' in question_lower or 'hp' in question_lower or 'kontak' in question_lower:
            if phones:
                phone_list = ", ".join(set(phones))
                return f"Berdasarkan {source}:\n\n{phone_list}\n\nKontak lengkap:\n{content}"
        
        if 'email' in question_lower:
            if emails:
                email_list = ", ".join(set(emails))
                return f"Berdasarkan {source}:\n\nEmail: {email_list}"
        
        # Fallback: return relevant snippet
        snippet = self._extract_relevant_snippet(content, question)
        return f"Berdasarkan {source}:\n\n{snippet}"
        
    def _is_garbled_output(self, text: str) -> bool:
        """Check if output looks garbled/reversed or unhelpful"""
        if not text or len(text) < 10:
            return True
        
        text_lower = text.lower()
        
        # Check for common garbled/unhelpful patterns from flan-t5
        unhelpful_patterns = [
            'pertanyaan pertanyaan',
            'berdasar konteks',
            'jawab pertanyaan',
            'answer the question',
            'based on context',
            'tidak ada informasi',
            'no information',
            'context:',
            'question:',
            'yang bersama yang bersama',  # repetitive garbled output
            'bersama yang bersama',
            'yang berkata terjadi',  # another garbled pattern
            'berkata tidak berkata',
            'data pertanyaan tahun',
            'data pertanyaan tersebut',
            'format yang bersah',
        ]
        
        for pattern in unhelpful_patterns:
            if pattern in text_lower:
                return True
        
        # Check for repetitive patterns (same word repeated 5+ times)
        import re
        words = text_lower.split()
        if len(words) > 5:
            from collections import Counter
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)
            if most_common and most_common[0][1] >= 5 and most_common[0][1] / len(words) > 0.3:
                return True
        
        # Check for reversed text patterns (consonant clusters that don't make sense)
        consonant_pattern = re.compile(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}')
        garbled_words = sum(1 for w in words if consonant_pattern.search(w))
        
        if len(words) > 0 and garbled_words / len(words) > 0.3:
            return True
        
        # If answer is too short and doesn't contain useful info
        if len(words) < 5:
            return True
        
        # Check if answer doesn't contain any expected content (names, data fields)
        # If context had database results but answer has none of the field values, likely garbled
        expected_markers = ['id:', 'name:', 'email:', '@', 'department:', 'position:', 'phone:']
        has_expected = any(marker in text_lower for marker in expected_markers)
        
        # If text looks like repeated gibberish without any data markers
        if not has_expected and 'yang' in text_lower and text_lower.count('yang') >= 3:
            return True
        
        return False
    
    def _generate_fallback_answer(self, hybrid_results: Dict[str, Any], question: str) -> str:
        """Generate a simpler fallback answer when LLM produces garbage"""
        pdf_docs = hybrid_results.get('pdf_documents', [])
        db_results = hybrid_results.get('database_results', {})
        chat_docs = hybrid_results.get('chat_documents', [])
        question_lower = question.lower()
        
        # Extract keywords from question
        keywords = []
        for word in question_lower.split():
            if len(word) > 3:
                keywords.append(word)
        
        # Priority 1: Database results (most structured)
        if db_results:
            for table_name, db_result in db_results.items():
                if db_result.record_count > 0:
                    # Format database results nicely
                    records_text = []
                    for record in db_result.data[:3]:  # Limit to 3 records
                        # Filter out internal fields
                        display_fields = {k: v for k, v in record.items() 
                                         if not k.startswith('_') and k not in ['search_vector', 'relevance_score', 'created_at']}
                        record_str = ", ".join(f"{k}: {v}" for k, v in display_fields.items())
                        records_text.append(f"• {record_str}")
                    
                    result_text = "\n".join(records_text)
                    return f"Berdasarkan data dari tabel {table_name}:\n\n{result_text}"
        
        # Priority 2: PDF documents
        if pdf_docs:
            # Find the most relevant document based on question keywords
            best_doc = None
            best_score = 0
            
            for doc in pdf_docs:
                content_lower = doc.page_content.lower()
                # Count how many question keywords appear in the content
                matches = sum(1 for kw in keywords if kw in content_lower)
                doc_score = doc.metadata.get('similarity_score', 0) + (matches * 0.1)
                
                if doc_score > best_score:
                    best_score = doc_score
                    best_doc = doc
            
            if best_doc is None:
                best_doc = pdf_docs[0]
            
            source = best_doc.metadata.get('source', 'dokumen')
            page = best_doc.metadata.get('page', '?')
            
            # Try to extract relevant sentence containing keyword
            content = best_doc.page_content
            # Convert keywords list to question string for _extract_relevant_snippet
            keyword_query = " ".join(keywords)
            relevant_snippet = self._extract_relevant_snippet(content, keyword_query)
            
            if relevant_snippet:
                return f"Berdasarkan {source} (halaman {page}):\n\n{relevant_snippet}"
            else:
                return f"Berdasarkan {source} (halaman {page}):\n\n{content}"
        
        # Priority 3: Chat results
        if chat_docs:
            best_chat = chat_docs[0]
            source = best_chat.metadata.get('source', 'chat')
            platform = best_chat.metadata.get('platform', 'unknown')
            return f"Berdasarkan percakapan dari {source} ({platform}):\n\n{best_chat.page_content}"
        
        return "Maaf, sistem tidak dapat menghasilkan jawaban yang valid. Silakan coba pertanyaan yang lebih spesifik."
    
    def get_source_info(self, hybrid_results: Dict[str, Any]) -> List[SourceInfo]:
        """Extract source information for response"""
        sources = []

        #PDF sources
        for doc in hybrid_results.get('pdf_documents', []):
            source_info = SourceInfo(
                type="pdf",
                source=doc.metadata.get('source', 'Unknown'),
                confidence=doc.metadata.get('similarity_score', 0),
                preview=self.truncate_context(doc.page_content, max_tokens=50),
                metadata={
                    'page': doc.metadata.get('page', 'Unknown'),
                    'collection_id': doc.metadata.get('collection_id', 'Unknown')
                }
            )

            sources.append(source_info)

         # Database sources
        db_results = hybrid_results.get('database_results', {})
        for table_name, db_result in db_results.items():
            if db_result.record_count > 0:
                source_info = SourceInfo(
                    type="database",
                    source=f"Table: {table_name}",
                    confidence=1.0,
                    preview=f"Menemukan {db_result.reord_count} record",
                    metadata={
                        'table': table_name,
                        'record_count': db_result.record_count
                    }
                )

                sources.append(source_info)

        return sources
        

        
# Global processor instance
processor = PDFQAProcessor()
