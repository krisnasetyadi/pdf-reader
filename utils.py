import os
from typing import List
import torch
from tqdm import tqdm
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import config
import logging

logger = logging.getLogger(__name__)


def process_pdfs(pdf_files: List[str], collection_id: str) -> int:
    """Process and index PDF files"""
    try:
        documents = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                pdf_reader = PdfReader(pdf_file)
                base_name = os.path.basename(pdf_file)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": base_name,
                                "page": page_num,
                                "collection_id": collection_id
                            }
                        )
                        documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing {pdf_file}: {str(e)}")
                continue
        if not documents:
            raise ValueError("No valid text content extracted from PDFs")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        )
        index_path = os.path.join(
            config.index_folder, f"{collection_id}.faiss"
        )
        if os.path.exists(index_path):
            vector_store = FAISS.load_local(
                os.path.join(config.index_folder, collection_id),
                embeddings
            )
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)
        save_path = os.path.join(config.index_folder, collection_id)
        vector_store.save_local(save_path)
        return len(documents)
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise
