# router/query.py
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging
import asyncio
from datetime import datetime

from processor import processor
from config import config
from models import QueryRequest, QAResponse
from langchain.schema import Document

router = APIRouter()
logger = logging.getLogger(__name__)


def truncate_document_content(docs: List[Document], max_chars: int = 800) -> List[Document]:
    truncated = []
    for d in docs:
        text = d.page_content or ""
        clipped = text[:max_chars] + ("..." if len(text) > max_chars else "")
        truncated.append(Document(page_content=clipped, metadata=d.metadata))
    return truncated


async def search_collection_async(collection_id: str, query: str, top_k: int):
    """
    Run blocking FAISS search in thread, return list of tuples (Document, score).
    """
    def _search():
        vs = processor.load_vector_store(collection_id)
        if not vs:
            return []
        # Use similarity_search_with_score if available
        try:
            results = vs.similarity_search_with_score(query, k=top_k)
        except AttributeError:
            # fallback to similarity_search (no scores) -> treat score=0
            docs = vs.similarity_search(query, k=top_k)
            results = [(doc, 0.0) for doc in docs]
        # attach collection metadata
        out = []
        for doc, score in results:
            if not isinstance(doc, Document):
                # langchain might return dict-like; convert if needed
                pass
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["collection_id"] = collection_id
            out.append((doc, float(score)))
        return out

    return await asyncio.to_thread(_search)


def merge_and_rank(scored_docs: List[tuple], top_n: int):
    # scored_docs: list of (Document, score)
    if not scored_docs:
        return []
    # FAISS distance: smaller = more similar (depending on implementation). If your scores are distances, sort ascending.
    # If your scores are similarity (higher better), sort descending.
    # We attempt to detect: if mean score > 1.0 assume similarity (descending), else distance (ascending).
    scores = [s for (_, s) in scored_docs]
    if not scores:
        return [d for (d, _) in scored_docs][:top_n]
    mean_score = sum(scores) / len(scores)
    if mean_score > 1.0:
        # similarity: high->low
        scored_docs.sort(key=lambda x: x[1], reverse=True)
    else:
        scored_docs.sort(key=lambda x: x[1])  # distance: low->high
    top_docs = [d for d, _ in scored_docs[:top_n]]
    return top_docs


@router.post("/query", response_model=QAResponse)
async def multi_source_query(request: QueryRequest):
    """
    Query across all collections.
    """
    start_time = datetime.now()
    try:

        collection_ids = processor.get_all_collections()

        if not collection_ids:
            raise HTTPException(status_code=404, detail="No collections found")

        # run searches in parallel (each will run FAISS in thread)
        tasks = [search_collection_async(cid, request.question,
                                         request.k_per_collection
                                         if hasattr(
                                             request, 'k_per_collection'
                                         )
                                         else config.k_per_collection)
                 for cid in collection_ids]
        results_nested = await asyncio.gather(*tasks)

        # flatten
        scored_docs = [item for sub in results_nested for item in sub]
        if not scored_docs:
            raise HTTPException(
                status_code=404, detail="No relevant documents found")

        # global rank
        top_docs = merge_and_rank(scored_docs, config.total_k_results)

        # truncate content to keep LLM prompt small
        truncated = truncate_document_content(top_docs, max_chars=800)

        # build temp vector store (blocking) in thread
        def _build_temp():
            return processor.build_temp_vector_store_from_docs(truncated)
        temp_vs = await asyncio.to_thread(_build_temp)

        # call LLM QA blocking in thread
        def _call_llm():
            retriever = temp_vs.as_retriever(
                search_kwargs={"k": min(len(truncated), config.k_results)})
            return processor.llm_qa_sync(retriever, request.question, return_source_documents=request.include_sources)

        llm_result = await asyncio.to_thread(_call_llm)

        # process sources if any
        sources = []
        if request.include_sources and llm_result.get("source_documents"):
            unique = set()
            for d in llm_result["source_documents"]:
                meta = d.metadata or {}
                info = f"[{meta.get('collection_id', 'unknown')}] {meta.get('source', 'unknown')} (page {meta.get('page', 'N/A')})"
                unique.add(info)
            sources = list(unique)

        processing_time = (datetime.now() - start_time).total_seconds()
        return QAResponse(
            answer=llm_result.get("result", ""),
            sources=sources,
            collection_id=request.collection_id or "all_collections",
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
