from urllib import request
from fastapi import APIRouter, HTTPException
import logging
import asyncio
from datetime import datetime

from processor import processor
from config import config
from models import QueryRequest, QAResponse, SearchType

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QAResponse)
async def query_documents(request: QueryRequest):
    """Enhanced query with better error handling"""
    start_time = datetime.now()

    try:
        #  validate the question
        if not request.question.strip() or len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Pertanyaan terlalu pendek atau kosong"
            )

        if not hasattr(request, 'search_type') or request.search_type == SearchType.UNSTRUCTURED:

        # determine which collections to search
        # if request.collection_id:
        #     if request.collection_id not in processor.get_all_collections():
        #         raise HTTPException(
        #             status_code=404,
        #             detail=f"Collection {request.collection_id} tidak ditemukan"
        #         )
        #     collection_ids = [request.collection_id]
        # else:
        #     collection_ids = processor.get_all_collections()

            collection_ids = ([request.collection_id] if request.collection_id 
                                else processor.get_all_collections())
            if not collection_ids:
                raise HTTPException(
                    status_code=404,
                    detail="Tidak ada koleksi dokumen yang tersedia"
                )

            # do search across collections
            relevant_docs = processor.search_across_collections(
                request.question,
                collection_ids,
                top_k=config.k_per_collection
            )

            if not relevant_docs:
                raise HTTPException(
                    status_code=404,
                    detail="Tidak ditemukan informasi relevan dalam dokumen"
                )

            # generate the answer
            answer = await asyncio.to_thread(
                processor.generate_answer, relevant_docs, request.question
            )

            # preparing sources with better formatting
            sources = []
            for doc in relevant_docs:
                source_info = f"{doc.metadata.get('source', 'Unknown')}"
                if 'page' in doc.metadata:
                    source_info += f" (Halaman {doc.metadata['page']})"
                sources.append(source_info)

            processing_time = (datetime.now() - start_time).total_seconds()

            return QAResponse(
                answer=answer,
                sources=sources,
                collection_id=request.collection_id or "all_collections",
                processing_time=processing_time
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Gunakan endpoint /query/hybrid untuk structured atau hybrid search"      )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan internal dalam pemrosesan query"
        )


@router.post("/query/debug")
async def query_debug(request: QueryRequest):
    """Debug endpoint to see search results"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Pertanyaan kosong")

        collection_ids = [request.collection_id] if request.collection_id else processor.get_all_collections()

        if not collection_ids:
            return {"error": "No collections available", "results": []}

        # Get search results
        relevant_docs = processor.search_across_collections(
            request.question, collection_ids, top_k=config.k_per_collection
        )

        # Format debug information
        debug_results = []
        for i, doc in enumerate(relevant_docs):
            debug_results.append({
                "rank": i + 1,
                "score": doc.metadata.get("similarity_score", 0),
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "collection": doc.metadata.get("collection_id", "Unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })

        return {
            "question": request.question,
            "total_results": len(relevant_docs),
            "results": debug_results
        }

    except Exception as e:
        logger.error(f"Debug query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
