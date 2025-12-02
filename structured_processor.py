# structured_processor.py
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from models import QueryIntent
from database import db_manager

logger = logging.getLogger(__name__)

class StructuredDataProcessor:
    def __init__(self):
        self.db_manager = db_manager
        
        # Define table schemas for better query understanding
        self.table_schemas = {
            'user_profiles': {
                'columns': ['id', 'name', 'email', 'department', 'position', 'phone', 'created_at'],
                'description': 'Data karyawan dan profil pengguna'
            },
            'products': {
                'columns': ['id', 'name', 'category', 'price', 'description', 'stock_quantity', 'created_at'],
                'description': 'Data produk dan inventory'
            },
            'orders': {
                'columns': ['id', 'user_id', 'product_id', 'quantity', 'total_amount', 'status', 'order_date', 'created_at'],
                'description': 'Data pesanan dan transaksi'
            }
        }
        
        # Intent patterns
        self.intent_patterns = {
            QueryIntent.COUNT: [
                r'(berapa|berapa banyak|jumlah|total|hitung).*(user|pengguna|karyawan|produk|barang|order|pesanan)',
                r'(total|jumlah).*(semua|semua data)',
                r'berapa.*total',
                r'count|total|jumlah'
            ],
            QueryIntent.SEARCH: [
                r'(cari|temukan|carikan|lihat).*(user|pengguna|karyawan|produk|barang|order|pesanan)',
                r'(user|pengguna|karyawan|produk|barang).*(bernama|dengan nama|yang namanya)',
                r'.*(dimana|mana).*',
                r'search|find|lookup'
            ],
            QueryIntent.LIST: [
                r'(tampilkan|lihat|show|list|daftar).*(semua|semua data|data)',
                r'semua.*(user|pengguna|karyawan|produk|barang|order|pesanan)',
                r'list|show all|display'
            ],
            QueryIntent.AGGREGATE: [
                r'(rata-rata|average|rerata|harga tertinggi|harga terendah|max|min)',
                r'(total|jumlah).*(harga|price|quantity)',
                r'(berapa).*(harga|price)',
                r'aggregate|sum|avg|max|min'
            ]
        }

    def analyze_intent(self, question: str) -> Tuple[QueryIntent, str, Dict[str, Any]]:
        """Analyze question to determine intent and target table"""
        question_lower = question.lower()
        
        # Determine intent
        intent = QueryIntent.UNKNOWN
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    intent = intent_type
                    break
            if intent != QueryIntent.UNKNOWN:
                break
        
        # Determine target table
        table_name = None
        table_keywords = {
            'user_profiles': ['user', 'pengguna', 'karyawan', 'profil', 'staff', 'employee'],
            'products': ['produk', 'barang', 'item', 'inventory', 'product'],
            'orders': ['order', 'pesanan', 'transaksi', 'pembelian', 'order']
        }
        
        for table, keywords in table_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                table_name = table
                break
        
        # Extract search terms and filters
        filters = self.extract_filters(question_lower)
        
        return intent, table_name, filters

    def extract_filters(self, question: str) -> Dict[str, Any]:
        """Extract filters from question"""
        filters = {}
        question_lower = question.lower()
        
        # Department filter
        departments = ['it', 'hr', 'finance', 'marketing']
        for dept in departments:
            if dept in question_lower:
                filters['department'] = dept
                break
        
        # Status filter for orders
        statuses = ['completed', 'pending', 'shipped', 'cancelled']
        for status in statuses:
            if status in question_lower:
                filters['status'] = status
                break
        
        # Price range filters
        price_patterns = [
            (r'harga.*(\d+).*sampai.*(\d+)', 'between'),
            (r'harga.*dibawah.*(\d+)', 'lt'),
            (r'harga.*diatas.*(\d+)', 'gt'),
            (r'harga.*kurang.*dari.*(\d+)', 'lt'),
            (r'harga.*lebih.*dari.*(\d+)', 'gt')
        ]
        
        for pattern, operator in price_patterns:
            match = re.search(pattern, question_lower)
            if match:
                if operator == 'between':
                    filters['price'] = {'operator': 'between', 'values': [int(match.group(1)), int(match.group(2))]}
                else:
                    filters['price'] = {'operator': operator, 'value': int(match.group(1))}
                break
        
        return filters

    def execute_structured_query(self, question: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute structured query based on natural language question"""
        start_time = datetime.now()
        
        try:
            # Analyze the question
            intent, detected_table, filters = self.analyze_intent(question)
            
            # Use provided table_name or detected table
            target_table = table_name or detected_table
            
            logger.info(f"Query analysis - Intent: {intent}, Table: {target_table}, Filters: {filters}")
            
            # Execute based on intent
            if intent == QueryIntent.COUNT:
                result = self.execute_count_query(target_table, filters)
            elif intent == QueryIntent.LIST:
                result = self.execute_list_query(target_table, filters)
            elif intent == QueryIntent.AGGREGATE:
                result = self.execute_aggregate_query(question, target_table, filters)
            else:  # SEARCH or UNKNOWN
                result = self.execute_search_query(question, target_table, filters)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                **result,
                "intent": intent,
                "table_used": target_table or "multiple",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Structured query execution failed: {str(e)}")
            return {
                "answer": f"Maaf, terjadi kesalahan dalam memproses query: {str(e)}",
                "data": [],
                "intent": QueryIntent.UNKNOWN,
                "table_used": "unknown",
                "processing_time": 0
            }

    def execute_count_query(self, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute count query"""
        try:
            if table_name:
                # Count specific table
                where_clause, params = self.build_where_clause(filters)
                query = f"SELECT COUNT(*) as count FROM {table_name} {where_clause}"
                result = db_manager.execute_query(query, params)
                count = result[0]['count'] if result else 0
                
                answer = f"Total {self.get_table_display_name(table_name)}: {count}"
                if filters:
                    filter_desc = self.describe_filters(filters)
                    answer += f" dengan filter {filter_desc}"
                    
                return {
                    "answer": answer,
                    "data": [{"count": count}],
                    "sql_query": query
                }
            else:
                # Count all tables
                counts = {}
                for table in self.table_schemas.keys():
                    query = f"SELECT COUNT(*) as count FROM {table}"
                    result = db_manager.execute_query(query)
                    counts[table] = result[0]['count'] if result else 0
                
                answer = "Total data dalam database:\n" + "\n".join(
                    [f"- {self.get_table_display_name(table)}: {count}" 
                     for table, count in counts.items()]
                )
                
                return {
                    "answer": answer,
                    "data": [counts],
                    "sql_query": "Multiple COUNT queries"
                }
                
        except Exception as e:
            logger.error(f"Count query failed: {str(e)}")
            return {
                "answer": f"Maaf, tidak dapat menghitung data: {str(e)}",
                "data": [],
                "sql_query": None
            }

    def execute_list_query(self, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list all records query"""
        try:
            if not table_name:
                return {
                    "answer": "Silakan sebutkan tabel yang ingin dilihat (user, produk, atau order)",
                    "data": [],
                    "sql_query": None
                }
            
            where_clause, params = self.build_where_clause(filters)
            query = f"SELECT * FROM {table_name} {where_clause} LIMIT 20"
            result = db_manager.execute_query(query, params)
            
            if not result:
                return {
                    "answer": f"Tidak ditemukan data dalam tabel {self.get_table_display_name(table_name)}",
                    "data": [],
                    "sql_query": query
                }
            
            answer = f"Data {self.get_table_display_name(table_name)}:"
            if filters:
                filter_desc = self.describe_filters(filters)
                answer += f" (filter: {filter_desc})"
            answer += f"\nDitemukan {len(result)} record."
            
            return {
                "answer": answer,
                "data": result,
                "sql_query": query
            }
            
        except Exception as e:
            logger.error(f"List query failed: {str(e)}")
            return {
                "answer": f"Maaf, tidak dapat mengambil data: {str(e)}",
                "data": [],
                "sql_query": None
            }

    def execute_search_query(self, question: str, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search query with text matching"""
        try:
            search_terms = self.extract_search_terms(question)
            
            if table_name:
                # Search in specific table
                where_conditions = []
                params = []
                
                # Add text search conditions
                schema = db_manager.get_table_schema(table_name)
                text_columns = [col['column_name'] for col in schema 
                              if col['data_type'] in ['character varying', 'text', 'varchar']]
                
                for column in text_columns:
                    for term in search_terms:
                        where_conditions.append(f"{column} ILIKE %s")
                        params.append(f"%{term}%")
                
                # Add filter conditions
                filter_conditions, filter_params = self.build_filter_conditions(filters, table_name)
                where_conditions.extend(filter_conditions)
                params.extend(filter_params)
                
                where_clause = " WHERE " + " OR ".join(where_conditions) if where_conditions else ""
                query = f"SELECT * FROM {table_name} {where_clause} LIMIT 10"
                result = db_manager.execute_query(query, params)
                
            else:
                # Search across all tables
                result = []
                for table in self.table_schemas.keys():
                    table_results = db_manager.search_in_table(table, search_terms, 5)
                    for item in table_results:
                        item['_table'] = table
                    result.extend(table_results)
            
            if not result:
                return {
                    "answer": "Tidak ditemukan data yang sesuai dengan pencarian.",
                    "data": [],
                    "sql_query": query if 'query' in locals() else "Multiple search queries"
                }
            
            answer = f"Ditemukan {len(result)} hasil pencarian."
            if search_terms:
                answer += f" Kata kunci: {', '.join(search_terms)}"
            
            return {
                "answer": answer,
                "data": result,
                "sql_query": query if 'query' in locals() else "Multiple search queries"
            }
            
        except Exception as e:
            logger.error(f"Search query failed: {str(e)}")
            return {
                "answer": f"Maaf, pencarian gagal: {str(e)}",
                "data": [],
                "sql_query": None
            }

    def execute_aggregate_query(self, question: str, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregate queries (avg, max, min, sum)"""
        try:
            question_lower = question.lower()
            
            if table_name == 'products' and any(word in question_lower for word in ['harga', 'price']):
                # Price-related aggregates
                where_clause, params = self.build_where_clause(filters)
                
                if 'rata' in question_lower or 'average' in question_lower:
                    query = f"SELECT AVG(price) as average_price FROM products {where_clause}"
                    result = db_manager.execute_query(query, params)
                    avg_price = result[0]['average_price'] if result else 0
                    return {
                        "answer": f"Rata-rata harga produk: {avg_price:,.2f}",
                        "data": [{"average_price": float(avg_price)}],
                        "sql_query": query
                    }
                elif 'tertinggi' in question_lower or 'max' in question_lower:
                    query = f"SELECT MAX(price) as max_price FROM products {where_clause}"
                    result = db_manager.execute_query(query, params)
                    max_price = result[0]['max_price'] if result else 0
                    return {
                        "answer": f"Harga produk tertinggi: {max_price:,.2f}",
                        "data": [{"max_price": float(max_price)}],
                        "sql_query": query
                    }
                elif 'terendah' in question_lower or 'min' in question_lower:
                    query = f"SELECT MIN(price) as min_price FROM products {where_clause}"
                    result = db_manager.execute_query(query, params)
                    min_price = result[0]['min_price'] if result else 0
                    return {
                        "answer": f"Harga produk terendah: {min_price:,.2f}",
                        "data": [{"min_price": float(min_price)}],
                        "sql_query": query
                    }
            
            return {
                "answer": "Fitur agregasi untuk query ini belum tersedia.",
                "data": [],
                "sql_query": None
            }
            
        except Exception as e:
            logger.error(f"Aggregate query failed: {str(e)}")
            return {
                "answer": f"Maaf, query agregasi gagal: {str(e)}",
                "data": [],
                "sql_query": None
            }

    def build_where_clause(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Build WHERE clause from filters"""
        if not filters:
            return "", []
        
        conditions = []
        params = []
        
        for key, value in filters.items():
            if key == 'department':
                conditions.append("department = %s")
                params.append(value)
            elif key == 'status':
                conditions.append("status = %s")
                params.append(value)
            elif key == 'price' and isinstance(value, dict):
                operator = value.get('operator')
                if operator == 'between':
                    conditions.append("price BETWEEN %s AND %s")
                    params.extend(value['values'])
                elif operator == 'lt':
                    conditions.append("price < %s")
                    params.append(value['value'])
                elif operator == 'gt':
                    conditions.append("price > %s")
                    params.append(value['value'])
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

    def build_filter_conditions(self, filters: Dict[str, Any], table_name: str) -> Tuple[List[str], List[Any]]:
        """Build filter conditions for search queries"""
        conditions = []
        params = []
        
        for key, value in filters.items():
            if key == 'department' and table_name == 'user_profiles':
                conditions.append("department = %s")
                params.append(value)
            elif key == 'status' and table_name == 'orders':
                conditions.append("status = %s")
                params.append(value)
        
        return conditions, params

    def extract_search_terms(self, question: str) -> List[str]:
        """Extract meaningful search terms from question"""
        stop_words = {'apa', 'siapa', 'dimana', 'kapan', 'berapa', 'bagaimana', 
                     'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'dalam', 'pada',
                     'data', 'user', 'cari', 'tampilkan', 'semua', 'lihat'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        meaningful_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return list(set(meaningful_terms))

    def get_table_display_name(self, table_name: str) -> str:
        """Get display name for table"""
        display_names = {
            'user_profiles': 'user',
            'products': 'produk', 
            'orders': 'order'
        }
        return display_names.get(table_name, table_name)

    def describe_filters(self, filters: Dict[str, Any]) -> str:
        """Describe filters in natural language"""
        descriptions = []
        
        for key, value in filters.items():
            if key == 'department':
                descriptions.append(f"department {value}")
            elif key == 'status':
                descriptions.append(f"status {value}")
            elif key == 'price':
                operator = value.get('operator')
                if operator == 'between':
                    descriptions.append(f"harga antara {value['values'][0]} dan {value['values'][1]}")
                elif operator == 'lt':
                    descriptions.append(f"harga dibawah {value['value']}")
                elif operator == 'gt':
                    descriptions.append(f"harga diatas {value['value']}")
        
        return ", ".join(descriptions)

# Global instance
structured_processor = StructuredDataProcessor()