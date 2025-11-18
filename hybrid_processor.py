"""
Hybrid Query Processor - Combines structured and unstructured data querying
"""
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, or_
import numpy as np

from database_sqlite import SessionLocal
from db_models import Document, DocumentChunk, BusinessEntity, QAPair, ChatMessage
from processor import processor
from config import config
import logging

logger = logging.getLogger(__name__)


class HybridQueryProcessor:
    """
    Processes queries across both structured (PostgreSQL) and unstructured (FAISS) data
    """

    def __init__(self):
        self.structured_keywords = {
            'entities', 'company', 'companies', 'business', 'organization',
            'who', 'when', 'where', 'list', 'count', 'total', 'sum'
        }

        self.unstructured_keywords = {
            'explain', 'how', 'why', 'what is', 'describe', 'tell me about',
            'definition', 'meaning', 'process', 'procedure'
        }

    def classify_query_intent(self, query: str) -> Dict[str, float]:
        """
        Classify query to determine if it needs structured, unstructured, or both
        Returns scores for each type (0-1)
        """
        query_lower = query.lower()

        structured_score = 0.0
        unstructured_score = 0.0

        # Check for structured data keywords
        for keyword in self.structured_keywords:
            if keyword in query_lower:
                structured_score += 0.3

        # Check for unstructured data keywords
        for keyword in self.unstructured_keywords:
            if keyword in query_lower:
                unstructured_score += 0.3

        # Check for question patterns
        if query_lower.startswith(('what is', 'how', 'why', 'explain')):
            unstructured_score += 0.4

        if query_lower.startswith(('list', 'show', 'find', 'count')):
            structured_score += 0.4

        # Check for specific entity mentions
        if re.search(r'\b(company|companies|business|organization)\b', query_lower):
            structured_score += 0.3

        # Normalize scores
        total = max(structured_score + unstructured_score, 1.0)

        return {
            'structured': min(structured_score / total, 1.0),
            'unstructured': min(unstructured_score / total, 1.0),
            'hybrid': min((structured_score + unstructured_score) / total, 1.0)
        }

    async def search_structured_data(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search structured data in PostgreSQL"""
        results = []

        try:
            with SessionLocal() as db:
                # Search business entities
                entities = db.query(BusinessEntity).filter(
                    or_(
                        BusinessEntity.name.ilike(f'%{query}%'),
                        BusinessEntity.description.ilike(f'%{query}%')
                    )
                ).limit(top_k).all()

                for entity in entities:
                    results.append({
                        'type': 'entity',
                        'source': 'postgresql',
                        'content': f"{entity.name}: {entity.description or ''}",
                        'metadata': {
                            'entity_id': entity.id,
                            'entity_type': entity.entity_type,
                            'attributes': entity.attributes
                        },
                        'relevance_score': 0.8  # Can be improved with better scoring
                    })

                # Search QA pairs
                qa_pairs = db.query(QAPair).filter(
                    or_(
                        QAPair.question.ilike(f'%{query}%'),
                        QAPair.answer.ilike(f'%{query}%')
                    )
                ).limit(top_k).all()

                for qa in qa_pairs:
                    results.append({
                        'type': 'qa_pair',
                        'source': 'postgresql',
                        'content': f"Q: {qa.question}\nA: {qa.answer}",
                        'metadata': {
                            'qa_id': qa.id,
                            'confidence': qa.confidence_score
                        },
                        'relevance_score': qa.confidence_score or 0.7
                    })

                # Search chat history
                chat_messages = db.query(ChatMessage).filter(
                    ChatMessage.content.ilike(f'%{query}%')
                ).limit(top_k).all()

                for msg in chat_messages:
                    results.append({
                        'type': 'chat_message',
                        'source': 'postgresql',
                        'content': msg.content,
                        'metadata': {
                            'message_id': msg.id,
                            'role': msg.role,
                            'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
                        },
                        'relevance_score': 0.6
                    })

        except Exception as e:
            logger.error(f"Error searching structured data: {e}")

        return results

    async def search_unstructured_data(self, query: str, collection_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Search unstructured data using existing FAISS processor"""
        results = []

        try:
            if not collection_ids:
                collection_ids = processor.get_all_collections()

            # Use existing processor search method
            docs = processor.search_across_collections(
                query,
                collection_ids,
                top_k=config.k_per_collection
            )

            for doc in docs:
                results.append({
                    'type': 'document_chunk',
                    'source': 'faiss',
                    'content': doc.page_content,
                    'metadata': {
                        'source': doc.metadata.get('source'),
                        'page': doc.metadata.get('page'),
                        'collection_id': doc.metadata.get('collection_id')
                    },
                    'relevance_score': doc.metadata.get('score', 0.7)
                })

        except Exception as e:
            logger.error(f"Error searching unstructured data: {e}")

        return results

    def fuse_results(self, structured_results: List[Dict], unstructured_results: List[Dict],
                     query_intent: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Combine and rank results from different sources based on relevance and intent
        """
        all_results = []

        # Weight results based on query intent
        for result in structured_results:
            weighted_score = result['relevance_score'] * query_intent['structured']
            result['final_score'] = weighted_score
            all_results.append(result)

        for result in unstructured_results:
            weighted_score = result['relevance_score'] * query_intent['unstructured']
            result['final_score'] = weighted_score
            all_results.append(result)

        # Sort by final score
        all_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Remove duplicates and limit results
        unique_results = []
        seen_content = set()

        for result in all_results:
            content_hash = hash(result['content'][:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                if len(unique_results) >= 10:  # Limit to top 10
                    break

        return unique_results

    async def hybrid_query(self, query: str, collection_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main hybrid query method that searches both structured and unstructured data
        """
        # Classify query intent
        intent = self.classify_query_intent(query)

        logger.info(f"Query intent scores: {intent}")

        # Search based on intent
        structured_results = []
        unstructured_results = []

        if intent['structured'] > 0.3:
            structured_results = await self.search_structured_data(query)

        if intent['unstructured'] > 0.3:
            unstructured_results = await self.search_unstructured_data(query, collection_ids)

        # Fuse and rank results
        fused_results = self.fuse_results(structured_results, unstructured_results, intent)

        return {
            'query': query,
            'intent_classification': intent,
            'structured_count': len(structured_results),
            'unstructured_count': len(unstructured_results),
            'fused_results': fused_results,
            'total_results': len(fused_results)
        }


# Global instance
hybrid_processor = HybridQueryProcessor()
