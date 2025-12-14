from config import config
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
import logging
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import threading
import torch
import os
from typing import List, Dict, Any, Optional
from models import SearchType, DatabaseResult, SourceInfo
from database import db_manager

logger = logging.getLogger(__name__)


class PDFQAProcessor:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store_cache = {}
        self._cache_lock = threading.RLock()
        self._initialized = False
        self._init_lock = threading.Lock()
        self.db_manager = None
        self._db_initialized = False

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
            'maker checker', 'enrich data', 'plte', 'bank indonesia'
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
                'stock', 'stok', 'kategori', 'category', 'jual', 'beli'
            ],
            "orders": [
                'order', 'pesanan', 'pembelian', 'transaksi', 'beli', 'pesan',
                'status', 'pending', 'completed', 'shipped', 'quantity',
                'total', 'amount', 'tanggal', 'invoice'
            ]
        }
        
        # Person name patterns (untuk deteksi nama orang tanpa hardcode)
        self.person_question_patterns = [
            'siapa', 'who is', 'nama', 'karyawan bernama', 'user bernama',
            'cari orang', 'find person', 'profile'
        ]

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

 #    # def initialize_components(self):
    #     """Initialize ML components with thread safety"""
    #     with self._init_lock:
    #         if self._initialized:
    #             return

    #         logger.info("Initializing NLP components...")
    #         try:
    #             # Initialize embeddings first (faster to load)
    #             self.embeddings = HuggingFaceEmbeddings(
    #                 model_name=config.embedding_model,
    #                 model_kwargs={
    #                     'device': 'cuda' if torch.cuda.is_available()
    #                     else 'cpu'
    #                 },
    #                 encode_kwargs={'normalize_embeddings': True}
    #             )

    #             # Initialize LLM components
    #             self.tokenizer = AutoTokenizer.from_pretrained(
    #                 config.model_name)

    #             # Add padding token if it doesn't exist
    #             if self.tokenizer.pad_token is None:
    #                 self.tokenizer.pad_token = self.tokenizer.eos_token

    #             model = AutoModelForSeq2SeqLM.from_pretrained(
    #                 config.model_name,
    #                 device_map="auto",
    #                 torch_dtype=torch.float16 if torch.cuda.is_available()
    #                 else torch.float32,
    #                 low_cpu_mem_usage=True
    #             )

    #             generation_config = GenerationConfig(
    #                 max_new_tokens=config.max_new_tokens,
    #                 temperature=config.temperature,
    #                 do_sample=True,
    #                 top_p=0.9,
    #                 repetition_penalty=1.1,
    #                 pad_token_id=self.tokenizer.pad_token_id
    #             )

    #             pipe = pipeline(
    #                 "text2text-generation",
    #                 model=model,
    #                 tokenizer=self.tokenizer,
    #                 generation_config=generation_config,
    #                 batch_size=4 if torch.cuda.is_available() else 1
    #             )

    #             self.llm = HuggingFacePipeline(pipeline=pipe)
    #             self._initialized = True
    #             logger.info("Components initialized successfully")

    #         except Exception as e:
    #             logger.error(f"Failed to initialize components: {str(e)}")
    #             raise

    def get_vector_store(self, collection_id):
        """Get vector store from cache or load from disk with thread safety"""
        with self._cache_lock:
            if collection_id in self.vector_store_cache:
                return self.vector_store_cache[collection_id]

            index_path = os.path.join(config.index_folder, collection_id)
            if not os.path.exists(index_path):
                logger.warning(f"Index path not found: {index_path}")
                return None

            try:
                # Check if index files exist
                if not all(os.path.exists(os.path.join(index_path, f))
                           for f in ["index.faiss", "index.pkl"]):
                    logger.error(f"Incomplete index files for {collection_id}")
                    return None

                vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vector_store_cache[collection_id] = vector_store
                return vector_store

            except Exception as e:
                logger.error(
                    f"Failed to load vector store for {collection_id}: {str(e)}"
                )
                return None

    def search_across_collections(self, query, collection_ids=None, top_k=5):
        """Enhanced search with query expansion and better scoring"""
        if collection_ids is None:
            collection_ids = self.get_all_collections()

        if not collection_ids:
            logger.warning("No collections available for search")
            return []

        # Expand query for better retrieval
        expanded_queries = self.expand_query(query)
        logger.info(f"Expanded queries: {expanded_queries}")

        all_results = []
        for expanded_query in expanded_queries:
            for collection_id in collection_ids:
                vector_store = self.get_vector_store(collection_id)
                if vector_store:
                    try:
                        # Try different search methods
                        try:
                            results = vector_store.similarity_search_with_relevance_scores(
                                expanded_query, k=top_k
                            )
                        except:
                            # Fallback for older versions
                            results_with_score = vector_store.similarity_search_with_score(
                                expanded_query, k=top_k
                            )
                            results = [(doc, 1 - score) for doc, score in results_with_score]

                        for doc, score in results:
                            if score > 0.5:  # Lower threshold for better recall
                                doc.metadata["collection_id"] = collection_id
                                doc.metadata["similarity_score"] = float(score)
                                doc.metadata["matched_query"] = expanded_query
                                all_results.append((doc, score))
                    except Exception as e:
                        logger.error(f"Search failed for {collection_id}: {str(e)}")
                        continue

        # Remove duplicates and sort by score
        unique_results = {}
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
            if content_hash not in unique_results or score > unique_results[content_hash][1]:
                unique_results[content_hash] = (doc, score)

        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_results[:config.total_k_results]]

    def truncate_context(self, text, max_tokens=500):
        """Truncate context to avoid token limit issues"""
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
            truncated_content = self.truncate_context(doc.page_content, max_tokens=200)

            context_parts.append(
                f"[Source: {source}, Page: {page}, Confidence: {score:.2f}]\n"
                f"{truncated_content}\n"
            )

        context = "\n".join(context_parts)

        # Further truncate the entire context if needed
        context = self.truncate_context(context, max_tokens=500)

        # Simplified prompt template to avoid instruction confusion
        prompt_template = """Berdasarkan informasi dari dokumen berikut, jawab pertanyaan dengan jelas dan akurat.

INFORMASI DOKUMEN:
{context}

PERTANYAAN: {question}

JAWABAN:"""

        prompt = prompt_template.format(context=context, question=question)

        try:
            # Use invoke() instead of __call__() to avoid deprecation warning
            result = self.llm.invoke(prompt)
            return result.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return "Maaf, terjadi kesalahan dalam menghasilkan jawaban."

    def get_all_collections(self):
        """Return list of available collections"""
        collections = []
        if not os.path.exists(config.index_folder):
            return collections

        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        return collections

    def invalidate_cache(self, collection_id=None):
        """Invalidate cache for specific collection or all"""
        with self._cache_lock:
            if collection_id:
                if collection_id in self.vector_store_cache:
                    del self.vector_store_cache[collection_id]
            else:
                self.vector_store_cache.clear()

    def initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_manager = db_manager
            self._db_initialized = True  # Fixed: use underscore prefix
            print('masuk db')
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db_initialized = False

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

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        config.model_name)
                except Exception as e:
                    logger.warning(f"Primary model {config.model_name} failed, using fallback: google/flan-t5-small")
                    config.model_name = "google/flan-t5-small"  # Fallback ke model yang lebih kecil
                    self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        config.model_name,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_available()
                        else torch.float32,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    logger.warning(f"Primary model {config.model_name} failed, using fallback: google/flan-t5-small")
                    config.model_name = "google/flan-t5-small"
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        config.model_name,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_available()
                        else torch.float32,
                        low_cpu_mem_usage=True
                    )
                generation_config = GenerationConfig(
                    max_new_tokens = config.max_new_tokens,
                    temperature = config.temperature,
                    do_sample = True,
                    top_p = 0.9,
                    repetition_penalty = 1.1,
                    pad_token_id = self.tokenizer.pad_token_id
                )

                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    generation_config=generation_config,
                    batch_size=4 if torch.cuda.is_available() else 1
                )

                self.llm = HuggingFacePipeline(pipeline=pipe)

                self.initialize_database()
                print('nyampe')
            
                self._initialized = True
                logger.info(f"All components initialized successfully with model: {config.model_name}")

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
        table_scores = {}
        
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

    def extract_search_terms(self, question: str) -> List[str]:
        """Extract meaningful search terms from question - NO hardcoded values"""
        import re
        
        # Stop words to filter out (common question words, not search-worthy)
        stop_words = {
            'apa', 'siapa', 'dimana', 'kapan', 'berapa', 'bagaimana', 'mengapa',
            'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'dalam', 'pada', 'untuk',
            'adalah', 'ini', 'itu', 'dengan', 'seperti', 'jika', 'maka',
            'cari', 'tampilkan', 'semua', 'lihat', 'tunjukkan', 'show',
            'find', 'search', 'get', 'the', 'what', 'who', 'where', 'when', 'how',
            'jumlah', 'total', 'hitung', 'count', 'berapa', 'banyak', 'orang'  # aggregation + generic words
        }
        
        # Column/field words that indicate we're looking for a value, not searching
        field_indicators = {'department', 'departemen', 'divisi', 'bagian', 'posisi', 'jabatan'}
        
        # Remove punctuation but keep alphanumeric and spaces
        cleaned = re.sub(r'[^\w\s]', ' ', question)
        words = cleaned.split()
        
        logger.info(f"🔍 Words from query: {words}")
        
        meaningful_terms = []
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip()
            
            # Skip stop words first
            if word_lower in stop_words:
                continue
            
            # If previous word was a field indicator, this word is likely a VALUE
            # Keep original case for proper nouns like "IT", "HR", "Finance"
            if i > 0 and words[i-1].lower() in field_indicators:
                # This is likely a value like "IT", "HR", "Finance"
                logger.info(f"🔍 Found value after field indicator: {word}")
                meaningful_terms.append(word.strip())  # Keep original case
                continue
            
            # Skip field indicators themselves
            if word_lower in field_indicators:
                continue
            
            # For short words (2 chars), only keep if they look like acronyms (all caps)
            if len(word_lower) <= 2:
                if word.isupper() and len(word) >= 2:
                    logger.info(f"🔍 Keeping acronym: {word}")
                    meaningful_terms.append(word)  # Keep "IT", "HR", etc.
                continue
                
            # Add other meaningful terms
            meaningful_terms.append(word_lower)
        
        unique_terms = list(set(meaningful_terms))
        
        logger.info(f"🔍 Extracted search terms: {unique_terms}")
        return unique_terms

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
            logger.info(f"Original terms: {search_terms} -> Expanded: {expanded_terms}")
            
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
        
        for term in search_terms:
            term_lower = term.lower()
            
            # Add synonyms from expansion dictionary
            for key, synonyms in self.db_query_expansion_terms.items():
                if key in term_lower or term_lower in synonyms:
                    expanded.add(key)
                    expanded.update(synonyms)
            
            # Apply simple Indonesian stemming
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

    def hybrid_search(self, question: str, collection_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform hybrid search across both structured and unstructured data"""

        analysis = self.analyze_question_type(question)
        search_terms = analysis["search_terms"]
        target_tables = analysis.get("target_tables", [])  # Get smart-routed tables

        pdf_docs = []
        if analysis["is_pdf_question"] or analysis["recommended_type"] == SearchType.HYBRID:
            pdf_docs = self.search_across_collections(
                question,
                collection_ids=collection_ids,
                top_k=config.k_per_collection
            )

        db_results = {}
        if analysis["is_db_question"] or analysis["recommended_type"] == SearchType.HYBRID:
            # Pass target_tables for smart routing
            db_results = self.query_structured_data(search_terms, target_tables)

        return {
            "pdf_documents": pdf_docs,
            "database_results": db_results,
            "search_analysis": analysis,
            "search_terms": search_terms,
            "target_tables": target_tables  # Include for debugging/transparency
        }

    def generate_hybrid_answer(self, hybrid_results: Dict[str, Any], question: str) -> str:
        """Generate answer combining both structured and unstructured data with relevance scoring"""
        pdf_docs = hybrid_results['pdf_documents']
        db_results = hybrid_results['database_results']
        target_tables = hybrid_results.get('target_tables', [])

        has_pdf_results = len(pdf_docs) > 0
        has_db_results = len(db_results) > 0
        
        if not has_pdf_results and not has_db_results:
            return "Maaf, tidak ditemukan informasi yang relevan baik dalam dokumen maupun database."

        # Prepare context from both sources
        context_parts = []

        # Add PDF context with confidence scores
        if has_pdf_results:
            context_parts.append("INFORMASI DARI DOKUMEN:")
            for i, doc in enumerate(pdf_docs[:3]): # limit to top 3 PDF results
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                score = doc.metadata.get('similarity_score', 0)
                truncated_content = self.truncate_context(doc.page_content, max_tokens=150)
                context_parts.append(f"[Dokumen: {source}, Halaman: {page}, Relevansi: {score:.2f}]\n{truncated_content}\n")

        # Add DB context with relevance scores
        if has_db_results:
            context_parts.append("INFORMASI DARI DATABASE:")
            context_parts.append(f"(Tabel yang dicari: {', '.join(target_tables)})")
            
            for table_name, db_result in db_results.items():
                if db_result.record_count > 0:
                    # Sort records by relevance_score if available
                    sorted_records = sorted(
                        db_result.data, 
                        key=lambda x: x.get('relevance_score', 0), 
                        reverse=True
                    )
                    
                    context_parts.append(f"\\nData dari tabel {table_name} ({db_result.record_count} record):")
                    for i, record in enumerate(sorted_records[:3]): # Limit to top 3 records
                        relevance = record.get('relevance_score', 'N/A')
                        # Filter out internal fields
                        display_fields = {k: v for k, v in record.items() 
                                         if not k.startswith('_') and k not in ['search_vector', 'relevance_score']}
                        record_str = ", ".join(f"{k}: {v}" for k, v in list(display_fields.items())[:5])
                        context_parts.append(f" - [Score: {relevance}] {record_str}")

                    if db_result.record_count > 3:
                        context_parts.append(f" - ... dan {db_result.record_count - 3} record lainnya")

        context = "\n".join(context_parts)
        context = self.truncate_context(context, max_tokens=500)

        # Enhanced prompt for hybrid answers
        prompt_template = """Berdasarkan informasi dari dokumen dan database berikut, jawab pertanyaan dengan jelas dan akurat. 
Sertakan informasi dari kedua sumber jika relevan. Prioritaskan informasi dengan skor relevansi tinggi.

{context}

PERTANYAAN: {question}

JAWABAN:"""

        prompt = prompt_template.format(context=context, question=question)

        try:
            result = self.llm.invoke(prompt)
            return result.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return "Maaf, terjadi kesalahan dalam menghasilkan jawaban."

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
