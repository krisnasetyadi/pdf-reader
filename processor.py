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
import threading
import torch
import os
from typing import List, Dict, Any, Optional, Tuple
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
        
        # Multi-LLM support
        self._llm_cache = {}  # Cache for loaded LLMs: {provider_model: llm_instance}
        self._current_provider = None
        self._current_model = None
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

    def get_vector_store(self, collection_id):
        """Get vector store from cache or load from disk with thread safety"""
        with self._cache_lock:
            if collection_id in self.vector_store_cache:
                logger.debug(f"Returning cached vector store for {collection_id}")
                return self.vector_store_cache[collection_id]

            logger.info(f"🔍 Loading vector store for collection: {collection_id}")
            index_path = os.path.join(config.index_folder, collection_id)
            logger.info(f"📁 Index path: {index_path}")
            
            if not os.path.exists(index_path):
                logger.warning(f"❌ Index path not found: {index_path}")
                return None

            # Check if index files exist
            faiss_file = os.path.join(index_path, "index.faiss")
            pkl_file = os.path.join(index_path, "index.pkl")
            logger.info(f"📄 FAISS file exists: {os.path.exists(faiss_file)}")
            logger.info(f"📄 PKL file exists: {os.path.exists(pkl_file)}")
            
            if not all(os.path.exists(os.path.join(index_path, f))
                       for f in ["index.faiss", "index.pkl"]):
                logger.error(f"❌ Incomplete index files for {collection_id}")
                return None

            # Check embeddings are loaded
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
                
                # Test the vector store with a simple search
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
                logger.error(f"🔍 Error type: {type(e).__name__}")
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
            for collection_id in collection_ids:
                logger.info(f"🔍 Searching collection {collection_id} with query: '{expanded_query}'")
                vector_store = self.get_vector_store(collection_id)
                if vector_store:
                    successful_collections += 1
                    try:
                        # Try different search methods
                        try:
                            results = vector_store.similarity_search_with_relevance_scores(
                                expanded_query, k=top_k
                            )
                            logger.info(f"📄 Found {len(results)} results with relevance scores")
                        except Exception as score_e:
                            logger.warning(f"⚠️ Relevance scores not supported, trying fallback: {score_e}")
                            # Fallback for older versions
                            results_with_score = vector_store.similarity_search_with_score(
                                expanded_query, k=top_k
                            )
                            results = [(doc, 1 - score) for doc, score in results_with_score]
                            logger.info(f"📄 Fallback search found {len(results)} results")

                        for doc, score in results:
                            logger.debug(f"📄 Result score: {score}, content preview: {doc.page_content[:50]}...")
                            if score > 0.05:  # Lower threshold for better recall (was 0.5, now 0.05)
                                doc.metadata["collection_id"] = collection_id
                                doc.metadata["similarity_score"] = float(score)
                                doc.metadata["matched_query"] = expanded_query
                                all_results.append((doc, score))
                                logger.debug(f"✅ Added result with score {score}")
                            else:
                                logger.debug(f"⏭️ Skipped result with low score {score}")
                    except Exception as e:
                        logger.error(f"❌ Search failed for {collection_id}: {str(e)}")
                        continue
                else:
                    logger.error(f"❌ Failed to load vector store for {collection_id}")

        logger.info(f"📊 Search summary: {successful_collections}/{len(collection_ids)} collections loaded, {len(all_results)} total results")

        # Remove duplicates and sort by score
        unique_results = {}
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
            if content_hash not in unique_results or score > unique_results[content_hash][1]:
                unique_results[content_hash] = (doc, score)

        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, score in sorted_results[:config.total_k_results]]
        
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
        """Return list of available PDF collections"""
        collections = []
        if not os.path.exists(config.index_folder):
            return collections

        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        return collections

    def get_all_chat_collections(self):
        """Return list of available chat collections"""
        collections = []
        if not os.path.exists(config.chat_index_folder):
            return collections

        for entry in os.listdir(config.chat_index_folder):
            entry_path = os.path.join(config.chat_index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        return collections

    def get_chat_vector_store(self, collection_id: str):
        """Get or load chat vector store from cache"""
        cache_key = f"chat_{collection_id}"
        
        with self._cache_lock:
            if cache_key in self.vector_store_cache:
                return self.vector_store_cache[cache_key]
        
        index_path = os.path.join(config.chat_index_folder, collection_id)
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

    def search_across_chat_collections(
        self, 
        question: str, 
        collection_ids: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List:
        """Search across chat collections"""
        if collection_ids is None:
            collection_ids = self.get_all_chat_collections()
        
        if not collection_ids:
            logger.info("No chat collections available")
            return []
        
        all_results = []
        
        for collection_id in collection_ids:
            vector_store = self.get_chat_vector_store(collection_id)
            if not vector_store:
                continue
            
            try:
                results = vector_store.similarity_search_with_relevance_scores(
                    question, k=top_k
                )
                
                for doc, score in results:
                    # Only include results above threshold
                    if score >= 0.3:  # Lower threshold for chat since context is important
                        doc.metadata['similarity_score'] = float(score)
                        doc.metadata['collection_id'] = collection_id
                        all_results.append(doc)
                        
            except Exception as e:
                logger.error(f"Error searching chat collection {collection_id}: {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)
        logger.info(f"📱 Found {len(all_results)} chat results")
        return all_results[:top_k * 2]  # Return more context for chats

    def invalidate_cache(self, collection_id=None):
        """Invalidate cache for specific collection or all"""
        with self._cache_lock:
            if collection_id:
                if collection_id in self.vector_store_cache:
                    del self.vector_store_cache[collection_id]
                # Also check chat cache
                chat_key = f"chat_{collection_id}"
                if chat_key in self.vector_store_cache:
                    del self.vector_store_cache[chat_key]
            else:
                self.vector_store_cache.clear()

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
        # Determine provider
        if provider:
            try:
                llm_provider = LLMProvider(provider.lower())
            except ValueError:
                logger.warning(f"Invalid provider '{provider}', using default")
                llm_provider = config.llm_provider
        else:
            llm_provider = config.llm_provider
        
        # Determine model based on provider
        if model:
            llm_model = model
        else:
            if llm_provider == LLMProvider.HUGGINGFACE:
                llm_model = config.model_name
            elif llm_provider == LLMProvider.GEMINI:
                llm_model = config.gemini_model
            else:
                llm_model = config.model_name
        
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
            # Fallback to default HuggingFace
            if llm_provider != LLMProvider.HUGGINGFACE:
                logger.info("⚠️ Falling back to HuggingFace default")
                return self.get_llm(LLMProvider.HUGGINGFACE.value, "google/flan-t5-base")
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
            raise ValueError("GEMINI_API_KEY not set in .env")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Install langchain-google-genai: pip install langchain-google-genai")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
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
        return f"{config.llm_provider.value}/{config.model_name}"

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
                if config.llm_provider == LLMProvider.HUGGINGFACE:
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

    def hybrid_search(
        self, 
        question: str, 
        collection_ids: Optional[List[str]] = None,
        include_chat: bool = True
    ) -> Dict[str, Any]:
        """Perform hybrid search across PDF, database, and chat data"""

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

        # NEW: Search chat collections
        chat_docs = []
        if include_chat:
            chat_docs = self.search_across_chat_collections(
                question,
                top_k=config.k_per_collection
            )

        return {
            "pdf_documents": pdf_docs,
            "database_results": db_results,
            "chat_documents": chat_docs,  # NEW
            "search_analysis": analysis,
            "search_terms": search_terms,
            "target_tables": target_tables  # Include for debugging/transparency
        }

    def generate_hybrid_answer(
        self, 
        hybrid_results: Dict[str, Any], 
        question: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate answer combining both structured and unstructured data with relevance scoring.
        
        Returns: (answer_text, model_identifier)
        """
        # Get LLM based on request parameters or defaults
        llm, model_id = self.get_llm(llm_provider, llm_model)
        
        pdf_docs = hybrid_results['pdf_documents']
        db_results = hybrid_results['database_results']
        target_tables = hybrid_results.get('target_tables', [])
        chat_docs = hybrid_results.get('chat_documents', [])  # NEW

        has_pdf_results = len(pdf_docs) > 0
        has_db_results = len(db_results) > 0
        has_chat_results = len(chat_docs) > 0  # NEW
        
        if not has_pdf_results and not has_db_results and not has_chat_results:
            return "Maaf, tidak ditemukan informasi yang relevan dalam dokumen, database, maupun chat logs.", model_id

        # Prepare context from all sources
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
            
            for table_name, db_result in db_results.items():
                if db_result.record_count > 0:
                    # Sort records by relevance_score if available
                    sorted_records = sorted(
                        db_result.data, 
                        key=lambda x: x.get('relevance_score', 0), 
                        reverse=True
                    )
                    
                    context_parts.append(f"\nData dari tabel {table_name}:")
                    for i, record in enumerate(sorted_records[:3]): # Limit to top 3 records
                        # Filter out internal fields - cleaner format for LLM
                        display_fields = {k: v for k, v in record.items() 
                                         if not k.startswith('_') and k not in ['search_vector', 'relevance_score', 'created_at']}
                        record_str = ", ".join(f"{k}: {v}" for k, v in list(display_fields.items())[:6])
                        context_parts.append(f"• {record_str}")

                    if db_result.record_count > 3:
                        context_parts.append(f"(dan {db_result.record_count - 3} record lainnya)")

   
        if has_chat_results:
            context_parts.append("\nINFORMASI DARI CHAT LOGS:")
            for i, doc in enumerate(chat_docs[:3]):  # limit to top 3 chat results
                source = doc.metadata.get('source', 'Unknown')
                platform = doc.metadata.get('platform', 'unknown')
                participants = doc.metadata.get('participants', '')
                score = doc.metadata.get('similarity_score', 0)
                time_start = doc.metadata.get('time_range_start', '')
                
                truncated_content = self.truncate_context(doc.page_content, max_tokens=200)
                context_parts.append(f"[Chat: {source}, Platform: {platform}, Relevansi: {score:.2f}]")
                if participants:
                    context_parts.append(f"Peserta: {participants}")
                context_parts.append(f"{truncated_content}\n")

        context = "\n".join(context_parts)
        context = self.truncate_context(context, max_tokens=400)  # Reduced for flan-t5 context limit

        # Extract main keyword from question for focused answering
        question_lower = question.lower()
        main_keyword = ""
        for keyword in ['buyback cash', 'buyback debt switch', 'lelang', 'auction', 'settlement', 'lpdu', 'sbn', 'sun']:
            if keyword in question_lower:
                main_keyword = keyword
                break
        
        # Improved prompt that focuses on the specific question
        if main_keyword:
            prompt_template = """Berikan definisi atau penjelasan tentang "{keyword}" berdasarkan konteks berikut.

Konteks:
{context}

Pertanyaan: {question}

Jawaban tentang {keyword}:"""
            prompt = prompt_template.format(keyword=main_keyword, context=context, question=question)
        else:
            # Check if this is primarily a database question
            if has_db_results and not has_pdf_results:
                prompt_template = """Berdasarkan data berikut, jawab pertanyaan dengan format yang jelas dan informatif.

Data:
{context}

Pertanyaan: {question}

Jawaban:"""
            else:
                prompt_template = """Jawab pertanyaan berikut berdasarkan konteks. Jawab dalam Bahasa Indonesia.

Konteks:
{context}

Pertanyaan: {question}

Jawaban:"""
            prompt = prompt_template.format(context=context, question=question)
        
        logger.debug(f"Generated prompt length: {len(prompt)} chars, using model: {model_id}")

        try:
            result = llm.invoke(prompt)
            
            # Handle different response types (ChatOllama returns AIMessage, HuggingFace returns str)
            if hasattr(result, 'content'):
                answer = result.content.strip()
            else:
                answer = str(result).strip()
            
            # Validate answer - if it looks garbled, return fallback
            if self._is_garbled_output(answer):
                logger.warning(f"Garbled output detected: {answer[:100]}")
                fallback = self._generate_fallback_answer(hybrid_results, question)
                return fallback, model_id
            
            return answer, model_id
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            fallback = self._generate_fallback_answer(hybrid_results, question)
            return fallback, model_id
    
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
            relevant_snippet = self._extract_relevant_snippet(content, keywords)
            
            if relevant_snippet:
                return f"Berdasarkan {source} (halaman {page}):\n\n{relevant_snippet}"
            else:
                content = self.truncate_context(content, max_tokens=150)
                return f"Berdasarkan {source} (halaman {page}):\n\n{content}"
        
        # Priority 3: Chat results
        if chat_docs:
            best_chat = chat_docs[0]
            source = best_chat.metadata.get('source', 'chat')
            platform = best_chat.metadata.get('platform', 'unknown')
            content = self.truncate_context(best_chat.page_content, max_tokens=200)
            return f"Berdasarkan percakapan dari {source} ({platform}):\n\n{content}"
        
        return "Maaf, sistem tidak dapat menghasilkan jawaban yang valid. Silakan coba pertanyaan yang lebih spesifik."
    
    def _extract_relevant_snippet(self, content: str, keywords: list) -> str:
        """Extract the most relevant sentence/paragraph containing keywords"""
        if not keywords:
            return ""
        
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]\s+', content)
        
        best_sentence = ""
        best_matches = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for kw in keywords if kw in sentence_lower)
            if matches > best_matches:
                best_matches = matches
                best_sentence = sentence.strip()
        
        if best_sentence and len(best_sentence) > 20:
            # Include a bit more context (up to 300 chars)
            return best_sentence[:300] + ("..." if len(best_sentence) > 300 else "")
        
        return ""

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
