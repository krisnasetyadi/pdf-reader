import pdfplumber
import os
import logging
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.schema import Document
from config import config

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with page information"""
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    # Clean text
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    # Add page information
                    full_text += f"{page_text}\n[PAGE {page_num}]\n\n"
        return full_text if full_text.strip() else None
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return None


def process_pdfs(pdf_paths, collection_id):
    """Process PDFs with improved text splitting"""
    documents = []

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if text:
            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "file_path": pdf_path,
                    "collection_id": collection_id
                }
            )
            documents.append(doc)

    if not documents:
        logger.error("No text could be extracted from any PDF")
        return 0

    # Improved text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        keep_separator=True
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        logger.error("No chunks created from documents")
        return 0

    # Create vector store
    try:
        from processor import processor
        vector_store = FAISS.from_documents(chunks, processor.embeddings)

        # Save vector store
        index_path = os.path.join(config.index_folder, collection_id)
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)

        logger.info(f"Created vector store with {len(chunks)} chunks for collection {collection_id}")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise
