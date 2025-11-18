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

        self.query_expansion_terms = {
            "apa itu": ["definisi", "pengertian", "arti", "makna", "jelaskan"],
            "proses": ["tahapan", "langkah", "mekanisme", "cara kerja"],
            "auction": ["lelang", "penawaran", "bidding", "tender"]
        }

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

    def initialize_components(self):
        """Initialize ML components with thread safety"""
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

                # Initialize LLM components dengan fallback
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        config.model_name)
                except Exception as e:
                    logger.warning(f"Primary model {config.model_name} failed, using fallback: google/flan-t5-small")
                    config.model_name = "google/flan-t5-small"  # Fallback ke model yang lebih kecil
                    self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

                # Add padding token if it doesn't exist
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
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    generation_config=generation_config,
                    batch_size=4 if torch.cuda.is_available() else 1
                )

                self.llm = HuggingFacePipeline(pipeline=pipe)
                self._initialized = True
                logger.info(f"Components initialized successfully with model: {config.model_name}")

            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                raise
    # def initialize_components(self):
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


# Global processor instance
processor = PDFQAProcessor()
