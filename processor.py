# processor.py
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
from langchain.prompts import PromptTemplate
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
        self._cache_lock = threading.Lock()

        self.definition_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.
        Jika pertanyaan dimulai dengan "Apa itu", berikan definisi singkat, jelas, dan padat dari istilah tersebut.
        Jika konteks tidak ditemukan, jawab "Informasi tidak tersedia."
        Konteks:
        {context}

        Pertanyaan:
        {question}

        Jawaban:
        """
        )

        # Prompt default untuk QA umum
        self.default_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        Gunakan informasi berikut untuk menjawab pertanyaan secara lengkap.
        Jika tidak ada jawaban di konteks, jawab: "Informasi tidak tersedia."
        Konteks:
        {context}

        Pertanyaan:
        {question}

        Jawaban:
        """
        )

    def choose_prompt(self, question: str):
        if question.strip().lower().startswith("apa itu"):
            return self.definition_prompt
        else:
            return self.default_prompt

    def initialize_components(self):
        """Initialize ML components"""
        logger.info("Initializing NLP components...")
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available()
                else torch.float32
            )
            generation_config = GenerationConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=True,
                top_p=0.95
            )
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=self.tokenizer,
                generation_config=generation_config
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            )
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def get_vector_store(self, collection_id):
        """Get vector store from cache or load from disk"""
        if collection_id not in self.vector_store_cache:
            index_path = os.path.join(config.index_folder, collection_id)
            if not os.path.exists(index_path):
                return None
            try:
                vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vector_store_cache[collection_id] = vector_store
            except Exception as e:
                logger.error(
                    f"Failed to load vector store for {collection_id}: {str(e)}")
                return None
        return self.vector_store_cache[collection_id]

    def invalidate_vector_store_cache(self, collection_id):
        """Remove vector store from cache"""
        if collection_id in self.vector_store_cache:
            del self.vector_store_cache[collection_id]

    def get_all_collections(self):
        # Kembalikan list ID koleksi yang ada
        return [name for name in os.listdir(self.index_folder) if os.path.isdir(os.path.join(self.index_folder, name))]

    def build_temp_vector_store(self, docs):
        # Buat vector store sementara dari list dokumen
        from langchain.vectorstores import FAISS
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        return FAISS.from_texts(texts, self.embedding, metadatas=metadatas)

    def llm_qa(self, retriever, query, include_sources=False):
        from langchain.chains import RetrievalQA
        prompt = self.choose_prompt(query)
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever.as_retriever(),
            return_source_documents=include_sources,
            chain_type_kwargs={"prompt": prompt}
        )
        return chain(query)

    def load_vector_store(self, collection_id):
        """Load vector store into cache if not present. Thread-safe."""
        with self._cache_lock:
            if collection_id in self.vector_store_cache:
                return self.vector_store_cache[collection_id]

            index_path = os.path.join(config.index_folder, collection_id)
            if not os.path.exists(index_path):
                return None

            try:
                vs = FAISS.load_local(
                    index_path, self.embeddings, allow_dangerous_deserialization=True)
                self.vector_store_cache[collection_id] = vs
                return vs
            except Exception as e:
                logger.error(
                    f"Failed to load vector store {collection_id}: {e}")
                return None

    def get_all_collections(self):
        """Return list of collection_ids (directories with index)"""
        collections = []
        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "index.faiss")):
                collections.append(entry)
        return collections

    def invalidate_vector_store_cache(self, collection_id):
        """Invalidate cache for a collection (already exists in your code)"""
        with self._cache_lock:
            if collection_id in self.vector_store_cache:
                del self.vector_store_cache[collection_id]

    def build_temp_vector_store_from_docs(self, docs):
        """Create in-memory FAISS from langchain Document list (docs already truncated)."""
        # FAISS.from_documents expects List[Document] and embeddings object
        try:
            temp_vs = FAISS.from_documents(docs, self.embeddings)
            return temp_vs
        except Exception as e:
            logger.error(f"Failed to build temporary vector store: {e}")
            raise

    def llm_qa_sync(self, retriever, query, return_source_documents=True):
        """Sync wrapper to call LLM via RetrievalQA (blocking)"""
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=return_source_documents
        )
        return qa_chain({"query": query})


processor = PDFQAProcessor()


# from config import config
# from transformers import (
#     AutoModelForSeq2SeqLM,
#     AutoTokenizer,
#     pipeline,
#     GenerationConfig
# )
# import logging
# from langchain.llms import HuggingFacePipeline
# import torch
# import os

# from langchain.vectorstores import FAISS

# logger = logging.getLogger(__name__)


# class PDFQAProcessor:
#     def __init__(self):
#         self.llm = None
#         self.tokenizer = None
#         self.embeddings = None
#         self.vector_store_cache = {}
#         self.initialize_components()

#     def initialize_components(self):
#         """Initialize ML components"""
#         logger.info("Initializing NLP components...")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
#             model = AutoModelForSeq2SeqLM.from_pretrained(
#                 config.model_name,
#                 device_map="auto",
#                 torch_dtype=torch.float16 if torch.cuda.is_available()
#                 else torch.float32
#             )
#             generation_config = GenerationConfig(
#                 max_new_tokens=config.max_new_tokens,
#                 temperature=config.temperature,
#                 do_sample=True,
#                 top_p=0.95
#             )
#             pipe = pipeline(
#                 "text2text-generation",
#                 model=model,
#                 tokenizer=self.tokenizer,
#                 generation_config=generation_config
#             )
#             self.llm = HuggingFacePipeline(pipeline=pipe)
#             logger.info("Components initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize components: {str(e)}")
#             raise

#     def get_vector_store(self, collection_id):
#         if collection_id not in self.vector_store_cache:
#             index_path = os.path.join(config.index_folder, collection_id)
#             self.vector_store_cache[collection_id] = FAISS.load_local(
#                 index_path,
#                 self.embeddings,
#                 allow_dangerous_deserialization=True
#             )
#         return self.vector_store_cache[collection_id]

# processor = PDFQAProcessor()
