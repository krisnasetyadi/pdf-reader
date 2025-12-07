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
        self.table_schemas = {}
        self.column_info = {}
        self._schema_initialized = False

        #patttern based on discovered schema
        self.intent_patterns = {}
        self.column_patterns = {}

        #initialize schema
        self.discover_schema()


    def discover_schema(self):
        #autho discover database schema and build pattern
        try:
            logger.info("Discover database schema ...")

            all_tables = self.db_manager.get_all_tables()
            logger.info(f"Discovered tables: {all_tables}")

            for table in all_tables:
                # get schema for each table
                schema = self.db_manager.get_table_schema(table)
                self.table_schemas[table] = {
                    'columns': [col['column_name'] for col in schema],
                    'column_details': schema,
                    'row_count': self.get_table_row_count(table),
                    'description': self.generate_table_description(table, schema)
                }

                self.build_column_patterns(table, schema)

            self.build_dynamic_patterns()
            self._schema_initialized = True
            logger.info(f'Schema discovery completed for {self.table_schemas} table')
                
        except Exception as e:
            logger.error(f"Schema discovery failed: {str(e)}")
            self._schema_initialized = False

    def get_table_row_count(self, table_name: str) -> int:
        "Get row count for a table"
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.db_manager.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {str(e)}")
            return 0

    def generate_table_description(self, table_name: str, schema: List[Dict[str, Any]]) -> str:
        "Generate description based on table name and columns"

        table_lower = table_name.lower()

        if any(word in table_lower for word in ['user', 'member', 'customer', 'employee']):
             base_desc = 'Data pengguna/anggota'
        elif any(word in table_lower for word in ['product', 'item', 'inventory', 'goods']):
             base_desc = 'Data produk/barang'
        elif any(word in table_lower for word in ['order', 'purchase', 'transaction', 'sale']):
            base_desc = 'Data tansaksi/penjualan'
        elif any (word in table_lower for word in ['document', 'file', 'pdf', 'report']):
            base_desc = 'Data dokumen/laporan'
        else:
            base_desc = f"Tabel {table_name}"

        key_columns = []

        for col in schema:
            col_name = col['column_name'].lower()
            if any(word in col_name for word in ['name', 'title', 'description']):
                key_columns.append(col['column_name'])
            elif col_name == 'id':
                key_columns.append(col['column_name'])

        if key_columns:
            base_desc += " dengan kolom utama: " + ", ".join(key_columns)

        return base_desc
    
    def build_column_patterns(self, table_name: str, schema: List[Dict]):
        """Build dynamic patterns for columns"""

        table_lower = table_name.lower()
        
        for col in schema:
            col_name = col['column_name']
            col_lower = col_name.lower()
            data_type = col['data_type']

            #storing column info
            self.column_info[f"{table_name}.{col_name}"] = {
                'table': table_name,
                'column': col_name,
                'data_type': data_type,
                'is_nullable': col['is_nullable']
            }

        # auto generate synonyms for common column names
        synonyms = self.generate_column_synonyms(col_name)

        # build patterns for text columns
        if data_type in ['character varying', 'text', 'varchar']:
            if any(word in col_lower for word in ['name', 'nama', 'title', 'judul']):
                self.column_patterns[col_name] = {
                    'type': 'name',
                    'synonyms': synonyms + ['name','nama', 'judul', 'title'],
                    'table': table_name
                }
            elif any(word in col_lower for word in ['description', 'deskripsi', 'keterangan', 'detail']):
                self.column_patterns[col_name] = {
                    'type': 'quantity',
                    'synonyms': synonyms + ['description', 'deskripsi', 'keterangan', 'detail'],
                    'table': table_name
                }
            elif any(word in col_lower for word in ['email', 'mail', 'surel']):
                self.column_patterns[col_name] = {
                    'type': 'email',
                    'synonyms': synonyms + ['email', 'mail', 'surel'],
                    'table': table_name
                }
            
            #build patterns for numeric columns
            elif data_type in ['integer', 'double precision', 'numeric', 'decimal', 'float']:
                if any(word in col_lower  for word in ['price', 'harga', 'amount', 'jumlah', 'total', 'tarif']):
                    self.column_patterns[col_name] = {
                        'type': 'price',
                        'synonyms': synonyms + ['price', 'harga', 'amount', 'jumlah', 'total', 'tarif'],
                        'table': table_name
                    }
            elif any(word in col_lower for word in ['quantity', 'qty', 'jumlah', 'kuantitas', 'stock', 'banyak']):
                self.column_patterns[col_name] = {
                    'type': 'quantity',
                    'synonyms': synonyms + ['quantity', 'qty', 'jumlah', 'kuantitas', 'stock', 'banyak'],
                    'table': table_name
                }

    def generate_column_synonyms(self, column_name: str) -> List[str]:
        """Generate synonyms for common column names"""

        col_lower = column_name.lower()
        synonyms  = [column_name, col_lower]

        # commons  patterns
        if '_' in column_name:
            #split snake case

            parts = column_name.split('_')
            synonyms.append(''.join(parts))
            synonyms.append(' '.join(parts))

        
        number_words = {'1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat', '5': 'lima',
                        'first': 'pertama', 'second': 'kedua', 'third': 'ketiga', 'fourth': 'keempat', 'fifth': 'kelima'}
        
        for num, word in number_words.items():
            if num in col_lower:
                synonyms.append(col_lower.replace(num, word))
        return list(set(synonyms))

    def build_dynamic_patterns(self):
        """Build dynamic intent patterns based on discovered schema"""

        # get table display names
        table_display_names = {}

        for table in self.table_schemas.keys():
            table_lower = table.lower()
            if 'user' in table_lower:
                table_display_names[table] = ['user', 'pengguna', 'karyawan', 'anggota']
            elif 'product' in table_lower:
                table_display_names[table] = ['produk', 'barang', 'item', 'goods']
            elif 'order' in table_lower:
                table_display_names[table] = ['order', 'pesanan', 'transaksi']
            else:
                parts = table.lower().split('_')
                table_display_names[table] = parts
        
        count_patterns = []
        for table, names in table_display_names.items():
            for name in names[:3]:
                count_patterns.extend([
                    rf'(berapa|berapa banyak|jumlah|total|hitung).*\b{name}\b',
                    rf'\b{ name}\b.*(berapa|berapa banyak|jumlah|total|hitung)',
                    rf'total.*\b{name}\b',
                    rf'jumlah.*\b{name}\b'
                ])

        search_patterns = []
        for table, names in table_display_names.items():
            for name in names[:2]:
                search_patterns.extend([
                    rf'(cari|temukan|lihat).*{name}',
                    rf'{name}*(bernama|dengan nama|namanya)',
                    rf'{name}.*(dimana|mana)'
                ])

        # build ilist paterns
        list_patterns = []
        for table, name in table_display_names.items():
            for name in names[:2]:
                list_patterns.extend([
                    rf'(tampilkan|lihat|daftar).*{name}',
                    rf'semua.*{name}',
                    rf'list.*{name}'
                ])

        self.intent_patterns = {
            QueryIntent.COUNT: count_patterns,
            QueryIntent.SEARCH: search_patterns,
            QueryIntent.LIST: list_patterns,
            QueryIntent.AGGREGATE: [
                f'(rata-rata|average|rerata|avg)',
                f'(tertinggi|terendah|paling tinggi|paling rendah|max|min)',
                f'(total|sum|jumlah).*(semua|all)'
            ]
        }
    
    def refresh_schema(self):
        """Refresh schema cache - call this when database changes"""

        logger.info("Refreshing database schema ...")

        self.table_schemas = {}
        self.column_info = {}
        self.discover_schema()


    def analyze_question_type(self, question: str) -> Dict[str, Any]:
        """Analyze question using dynamic patterns"""

        if not self._schema_initialized:
            self.discover_schema()

        question_lower = question.lower()

        # use dynamic patterns
        intent = QueryIntent.UNKNOWN
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    intent = intent_type
                    break
            if intent != QueryIntent.UNKNOWN:
                break

        target_table = None
        for table, schema in self.table_schemas.items():
            table_lower = table.lower()
            table_display = self.get_table_display_name(table)

            if any(word in question_lower for word in [table_lower, table_display]):
                target_table = table
                break

        if not target_table:
            for col_name, col_info in self.column_patterns.items():
                if any(syn in question_lower for syn in col_info['synonyms']):
                    target_table = col_info['table']
                    break

        search_terms = self.extract_search_terms(question)

        return {
            'recommended_type': self.determine_search_type(intent, target_table),
            'intent': intent,
            'target_table': target_table,
            'is_db_question': intent != QueryIntent.UNKNOWN or target_table is not None,
            'is_pdf_question': False,
            'search_terms': search_terms,
            'detected_columns': self.detect_columns_in_question(question)
        }
    
    def detect_columns_in_question(self, question:str) -> List[Dict[str, str]]:
        """Detect which columns are mentioned in the question"""
        detected = []
        question_lower = question.lower()

        for col_name, col_info in self.column_patterns.items():
            for synonym in col_info['synonyms']:
                if synonym in question_lower:
                    detected.append({
                        'column': col_name, 
                        'table': col_info['table'],
                        'type': col_info['type'],
                        'matched_synonym': synonym
                    })

                    break
        return detected

    def determine_search_type(self, intent: QueryIntent, target_table: Optional[str]) -> str:

        """Determine search type baed on intent and table"""

        from models import SearchType

        if intent == QueryIntent.UNKNOWN and not target_table:
            return SearchType.UNSTRUCTURED
        elif intent != QueryIntent.UNKNOWN and target_table:
            return SearchType.STRUCTURED
        else:
            return SearchType.HYBRID
        
    def get_table_display_name(self, table_name:str) -> str:
        """Get user-friendly table name"""

        table_lower = table_name.lower()

        # common mapping
        if 'user' in table_lower:
            return 'user'
        elif 'product' in table_lower:
            return 'produk'
        elif 'order' in table_lower:
            return 'order'

        # convert shnake case to readable

        return table_name.replace('_', ' ')

    def execute_structured_query(self, question: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute query with auto-discovered schema"""
        from models import QueryIntent
        if not self._schema_initialized:
            self.discover_schema()

        analysis = self.analyze_question_type(question)
        # if no table specified or detected, show available tables

        if not analysis['target_table'] and not table_name:
            return self.show_available_tables(question)

        target_table = table_name or analysis['target_table']

        # Execute based on intent
        if analysis['intent'] == QueryIntent.COUNT:
            return self.execute_count_query(target_table, analysis)
        elif analysis['intent'] == QueryIntent.LIST:
            return self.execute_list_query(target_table, analysis)
        elif analysis['intent'] == QueryIntent.AGGREGATE:
            return self.execute_aggregate_query(question, target_table, analysis)
        else:  # SEARCH or UNKNOWN
            return self.execute_search_query(question, target_table, analysis)

    # structured_processor.py - Update show_available_tables method
    def show_available_tables(self, question: str) -> Dict[str, Any]:
        """Show available tables when none specified"""
        from models import QueryIntent  # Add import
        
        table_list = []
        for table, schema in self.table_schemas.items():
            table_list.append({
                'name': table,
                'display_name': self.get_table_display_name(table),
                'description': schema['description'],
                'row_count': schema.get('row_count', 0),
                'columns': schema['columns'][:5]
            })
        
        answer = f"Saya menemukan {len(table_list)} tabel dalam database:\n\n"
        for table_info in table_list:
            answer += f"• {table_info['display_name']} ({table_info['name']})\n"
            answer += f"  {table_info['description']}\n"
            answer += f"  {table_info['row_count']} records, contoh kolom: {', '.join(table_info['columns'])}\n\n"
        
        answer += f"\nCoba tanyakan: 'total {table_list[0]['display_name']}' atau 'data {table_list[1]['display_name']}'"
        
        return {
            "answer": answer,
            "data": table_list,
            "intent": QueryIntent.SHOW_TABLES,  # Use enum value
            "table_used": "system",
            "sql_query": None,
            "processing_time": 0
        }
    # def show_available_tables(self, question: str) -> Dict[str, Any]:
    #     """Show available tables when none specified"""

    #     table_list = []
    #     for table, schema in self.table_schemas.items():
    #         table_list.append({
    #             'name': table,
    #             'display_name': self.get_table_display_name(table),
    #             'description': schema['description'],
    #             'row_count': schema.get('row_count', 0),
    #             'columns': schema['columns'][:5]
    #         })

    #     answer = f"saya menemukan {len(table_list)} tabel dalam database: \n\n"
    #     for table_info in table_list:
    #         answer += f"- {table_info['display_name']} ({table_info['row_count']})\n"
    #         answer += f"  Deskripsi: {table_info['description']}\n"
    #         answer += f" {table_info['row_count']} records, contoh kolom: {', '.join(table_info['columns'])}\n\n"

    #         answer += f"\n coba tanyakan: 'total {table_list[0]['display_name']} atau data {table_list[1]['display_name']}'"

    #     return {
    #      "answer": answer,
    #      "data": table_list,
    #      "intent": "show_tables",
    #      "table_used": "system",
    #      "sql_query": None,
    #      "processing_time": 0
    #     }


    #            # Define table schemas for better query understanding
    #     # self.table_schemas = {
    #     #     'user_profiles': {
    #     #         'columns': ['id', 'name', 'email', 'department', 'position', 'phone', 'created_at'],
    #     #         'description': 'Data karyawan dan profil pengguna'
    #     #     },
    #     #     'products': {
    #     #         'columns': ['id', 'name', 'category', 'price', 'description', 'stock_quantity', 'created_at'],
    #     #         'description': 'Data produk dan inventory'
    #     #     },
    #     #     'orders': {
    #     #         'columns': ['id', 'user_id', 'product_id', 'quantity', 'total_amount', 'status', 'order_date', 'created_at'],
    #     #         'description': 'Data pesanan dan transaksi'
    #     #     }
    #     # }
        
    #     # # Intent patterns
    #     # self.intent_patterns = {
    #     #     QueryIntent.COUNT: [
    #     #         r'(berapa|berapa banyak|jumlah|total|hitung).*(user|pengguna|karyawan|produk|barang|order|pesanan)',
    #     #         r'(total|jumlah).*(semua|semua data)',
    #     #         r'berapa.*total',
    #     #         r'count|total|jumlah'
    #     #     ],
    #     #     QueryIntent.SEARCH: [
    #     #         r'(cari|temukan|carikan|lihat).*(user|pengguna|karyawan|produk|barang|order|pesanan)',
    #     #         r'(user|pengguna|karyawan|produk|barang).*(bernama|dengan nama|yang namanya)',
    #     #         r'.*(dimana|mana).*',
    #     #         r'search|find|lookup'
    #     #     ],
    #     #     QueryIntent.LIST: [
    #     #         r'(tampilkan|lihat|show|list|daftar).*(semua|semua data|data)',
    #     #         r'semua.*(user|pengguna|karyawan|produk|barang|order|pesanan)',
    #     #         r'list|show all|display'
    #     #     ],
    #     #     QueryIntent.AGGREGATE: [
    #     #         r'(rata-rata|average|rerata|harga tertinggi|harga terendah|max|min)',
    #     #         r'(total|jumlah).*(harga|price|quantity)',
    #     #         r'(berapa).*(harga|price)',
    #     #         r'aggregate|sum|avg|max|min'
    #     #     ]
    #     # }

    # def analyze_intent(self, question: str) -> Tuple[QueryIntent, str, Dict[str, Any]]:
    #     """Analyze question to determine intent and target table"""
    #     question_lower = question.lower()
        
    #     # Determine intent
    #     intent = QueryIntent.UNKNOWN
    #     for intent_type, patterns in self.intent_patterns.items():
    #         for pattern in patterns:
    #             if re.search(pattern, question_lower):
    #                 intent = intent_type
    #                 break
    #         if intent != QueryIntent.UNKNOWN:
    #             break
        
    #     # Determine target table
    #     table_name = None
    #     table_keywords = {
    #         'user_profiles': ['user', 'pengguna', 'karyawan', 'profil', 'staff', 'employee'],
    #         'products': ['produk', 'barang', 'item', 'inventory', 'product'],
    #         'orders': ['order', 'pesanan', 'transaksi', 'pembelian', 'order']
    #     }
        
    #     for table, keywords in table_keywords.items():
    #         if any(keyword in question_lower for keyword in keywords):
    #             table_name = table
    #             break
        
    #     # Extract search terms and filters
    #     filters = self.extract_filters(question_lower)
        
    #     return intent, table_name, filters

    # def extract_filters(self, question: str) -> Dict[str, Any]:
    #     """Extract filters from question"""
    #     filters = {}
    #     question_lower = question.lower()
        
    #     # Department filter
    #     departments = ['it', 'hr', 'finance', 'marketing']
    #     for dept in departments:
    #         if dept in question_lower:
    #             filters['department'] = dept
    #             break
        
    #     # Status filter for orders
    #     statuses = ['completed', 'pending', 'shipped', 'cancelled']
    #     for status in statuses:
    #         if status in question_lower:
    #             filters['status'] = status
    #             break
        
    #     # Price range filters
    #     price_patterns = [
    #         (r'harga.*(\d+).*sampai.*(\d+)', 'between'),
    #         (r'harga.*dibawah.*(\d+)', 'lt'),
    #         (r'harga.*diatas.*(\d+)', 'gt'),
    #         (r'harga.*kurang.*dari.*(\d+)', 'lt'),
    #         (r'harga.*lebih.*dari.*(\d+)', 'gt')
    #     ]
        
    #     for pattern, operator in price_patterns:
    #         match = re.search(pattern, question_lower)
    #         if match:
    #             if operator == 'between':
    #                 filters['price'] = {'operator': 'between', 'values': [int(match.group(1)), int(match.group(2))]}
    #             else:
    #                 filters['price'] = {'operator': operator, 'value': int(match.group(1))}
    #             break
        
    #     return filters

    # def execute_structured_query(self, question: str, table_name: Optional[str] = None) -> Dict[str, Any]:
    #     """Execute structured query based on natural language question"""
    #     start_time = datetime.now()
        
    #     try:
    #         # Analyze the question
    #         intent, detected_table, filters = self.analyze_intent(question)
            
    #         # Use provided table_name or detected table
    #         target_table = table_name or detected_table
            
    #         logger.info(f"Query analysis - Intent: {intent}, Table: {target_table}, Filters: {filters}")
            
    #         # Execute based on intent
    #         if intent == QueryIntent.COUNT:
    #             result = self.execute_count_query(target_table, filters)
    #         elif intent == QueryIntent.LIST:
    #             result = self.execute_list_query(target_table, filters)
    #         elif intent == QueryIntent.AGGREGATE:
    #             result = self.execute_aggregate_query(question, target_table, filters)
    #         else:  # SEARCH or UNKNOWN
    #             result = self.execute_search_query(question, target_table, filters)
            
    #         processing_time = (datetime.now() - start_time).total_seconds()
            
    #         return {
    #             **result,
    #             "intent": intent,
    #             "table_used": target_table or "multiple",
    #             "processing_time": processing_time
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Structured query execution failed: {str(e)}")
    #         return {
    #             "answer": f"Maaf, terjadi kesalahan dalam memproses query: {str(e)}",
    #             "data": [],
    #             "intent": QueryIntent.UNKNOWN,
    #             "table_used": "unknown",
    #             "processing_time": 0
    #         }

    # def execute_count_query(self, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
    #     """Execute count query"""
    #     start_time = datetime.now()
    #     try:
    #         if table_name:
    #             # Count specific table
    #             where_clause, params = self.build_where_clause(filters)
    #             query = f"SELECT COUNT(*) as count FROM {table_name} {where_clause}"
    #             result = db_manager.execute_query(query, tuple(params))
    #             count = result[0]['count'] if result else 0
                
    #             answer = f"Total {self.get_table_display_name(table_name)}: {count}"
    #             if filters:
    #                 filter_desc = self.describe_filters(filters)
    #                 answer += f" dengan filter {filter_desc}"
                    
    #             return {
    #                 "answer": answer,
    #                 "data": [{"count": count}],
    #                 "intent": QueryIntent.COUNT,
    #                 "table_used": target_table,  # Add this
    #                 "sql_query": query,
    #                 "processing_time": processing_time
    #             }
    #         else:
    #             # Count all tables
    #             counts = {}
    #             for table in self.table_schemas.keys():
    #                 query = f"SELECT COUNT(*) as count FROM {table}"
    #                 result = db_manager.execute_query(query)
    #                 counts[table] = result[0]['count'] if result else 0
                
    #             answer = "Total data dalam database:\n" + "\n".join(
    #                 [f"- {self.get_table_display_name(table)}: {count}" 
    #                  for table, count in counts.items()]
    #             )
                
    #             return {
    #                 "answer": answer,
    #                 "data": [counts],
    #                 "sql_query": "Multiple COUNT queries"
    #             }
                
    #     except Exception as e:
    #         logger.error(f"Count query failed: {str(e)}")
    #         return {
    #             "answer": f"Maaf, tidak dapat menghitung data: {str(e)}",
    #             "data": [],
    #             "sql_query": None
    #         }

    # def execute_list_query(self, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
    #     """Execute list all records query"""
    #     try:
    #         if not table_name:
    #             return {
    #                 "answer": "Silakan sebutkan tabel yang ingin dilihat (user, produk, atau order)",
    #                 "data": [],
    #                 "sql_query": None
    #             }
            
    #         where_clause, params = self.build_where_clause(filters)
    #         query = f"SELECT * FROM {table_name} {where_clause} LIMIT 20"
    #         result = db_manager.execute_query(query, tuple(params))
            
    #         if not result:
    #             return {
    #                 "answer": f"Tidak ditemukan data dalam tabel {self.get_table_display_name(table_name)}",
    #                 "data": [],
    #                 "sql_query": query
    #             }
            
    #         answer = f"Data {self.get_table_display_name(table_name)}:"
    #         if filters:
    #             filter_desc = self.describe_filters(filters)
    #             answer += f" (filter: {filter_desc})"
    #         answer += f"\nDitemukan {len(result)} record."
            
    #         return {
    #             "answer": answer,
    #             "data": result,
    #             "sql_query": query
    #         }
            
    #     except Exception as e:
    #         logger.error(f"List query failed: {str(e)}")
    #         return {
    #             "answer": f"Maaf, tidak dapat mengambil data: {str(e)}",
    #             "data": [],
    #             "sql_query": None
    #         }

    # def execute_search_query(self, question: str, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
    #     """Execute search query with text matching"""
    #     try:
    #         search_terms = self.extract_search_terms(question)
            
    #         if table_name:
    #             # Search in specific table
    #             where_conditions = []
    #             params = []
                
    #             # Add text search conditions
    #             schema = db_manager.get_table_schema(table_name)
    #             text_columns = [col['column_name'] for col in schema 
    #                           if col['data_type'] in ['character varying', 'text', 'varchar']]
                
    #             for column in text_columns:
    #                 for term in search_terms:
    #                     where_conditions.append(f"{column} ILIKE %s")
    #                     params.append(f"%{term}%")
                
    #             # Add filter conditions
    #             filter_conditions, filter_params = self.build_filter_conditions(filters, table_name)
    #             where_conditions.extend(filter_conditions)
    #             params.extend(filter_params)
                
    #             where_clause = " WHERE " + " OR ".join(where_conditions) if where_conditions else ""
    #             query = f"SELECT * FROM {table_name} {where_clause} LIMIT 10"
    #             result = db_manager.execute_query(query,tuple(params))
                
    #         else:
    #             # Search across all tables
    #             result = []
    #             for table in self.table_schemas.keys():
    #                 table_results = db_manager.search_in_table(table, search_terms, 5)
    #                 for item in table_results:
    #                     item['_table'] = table
    #                 result.extend(table_results)
            
    #         if not result:
    #             return {
    #                 "answer": "Tidak ditemukan data yang sesuai dengan pencarian.",
    #                 "data": [],
    #                 "sql_query": query if 'query' in locals() else "Multiple search queries"
    #             }
            
    #         answer = f"Ditemukan {len(result)} hasil pencarian."
    #         if search_terms:
    #             answer += f" Kata kunci: {', '.join(search_terms)}"
            
    #         return {
    #             "answer": answer,
    #             "data": result,
    #             "sql_query": query if 'query' in locals() else "Multiple search queries"
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Search query failed: {str(e)}")
    #         return {
    #             "answer": f"Maaf, pencarian gagal: {str(e)}",
    #             "data": [],
    #             "sql_query": None
    #         }

    # def execute_aggregate_query(self, question: str, table_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
    #     """Execute aggregate queries (avg, max, min, sum)"""
    #     try:
    #         question_lower = question.lower()
            
    #         if table_name == 'products' and any(word in question_lower for word in ['harga', 'price']):
    #             # Price-related aggregates
    #             where_clause, params = self.build_where_clause(filters)
                
    #             if 'rata' in question_lower or 'average' in question_lower:
    #                 query = f"SELECT AVG(price) as average_price FROM products {where_clause}"
    #                 result = db_manager.execute_query(query, tuple(params))
    #                 avg_price = result[0]['average_price'] if result else 0
    #                 return {
    #                     "answer": f"Rata-rata harga produk: {avg_price:,.2f}",
    #                     "data": [{"average_price": float(avg_price)}],
    #                     "sql_query": query
    #                 }
    #             elif 'tertinggi' in question_lower or 'max' in question_lower:
    #                 query = f"SELECT MAX(price) as max_price FROM products {where_clause}"
    #                 result = db_manager.execute_query(query, tuple(params))
    #                 max_price = result[0]['max_price'] if result else 0
    #                 return {
    #                     "answer": f"Harga produk tertinggi: {max_price:,.2f}",
    #                     "data": [{"max_price": float(max_price)}],
    #                     "sql_query": query
    #                 }
    #             elif 'terendah' in question_lower or 'min' in question_lower:
    #                 query = f"SELECT MIN(price) as min_price FROM products {where_clause}"
    #                 result = db_manager.execute_query(query, tuple(params))
    #                 min_price = result[0]['min_price'] if result else 0
    #                 return {
    #                     "answer": f"Harga produk terendah: {min_price:,.2f}",
    #                     "data": [{"min_price": float(min_price)}],
    #                     "sql_query": query
    #                 }
            
    #         return {
    #             "answer": "Fitur agregasi untuk query ini belum tersedia.",
    #             "data": [],
    #             "sql_query": None
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Aggregate query failed: {str(e)}")
    #         return {
    #             "answer": f"Maaf, query agregasi gagal: {str(e)}",
    #             "data": [],
    #             "sql_query": None
    #         }

# structured_processor.py - Update semua execute methods

    def execute_count_query(self, target_table: str, analysis: Dict) -> Dict[str, Any]:
        """Execute count query dengan semua required fields"""
        start_time = datetime.now()  # Add timing
        
        try:
            where_clause, params = self.build_where_clause(analysis.get('filters', {}))
            query = f"SELECT COUNT(*) as count FROM {target_table} {where_clause}"
            result = db_manager.execute_query(query, params)
            count = result[0]['count'] if result else 0
            
            answer = f"Total {self.get_table_display_name(target_table)}: {count}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": answer,
                "data": [{"count": count}],
                "intent": QueryIntent.COUNT,
                "table_used": target_table,  # Add this
                "sql_query": query,
                "processing_time": processing_time  # Add this
            }
            
        except Exception as e:
            logger.error(f"Count query failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": f"Maaf, tidak dapat menghitung data: {str(e)}",
                "data": [],
                "intent": QueryIntent.UNKNOWN,
                "table_used": target_table,
                "sql_query": None,
                "processing_time": processing_time
            }

    def execute_search_query(self, question: str, target_table: str, analysis: Dict) -> Dict[str, Any]:
        """Execute search query dengan semua required fields"""
        start_time = datetime.now()  # Add timing
        
        try:
            search_terms = analysis.get('search_terms', [])
            filters = analysis.get('filters', {})
            
            # Build query
            where_conditions = []
            params = []
            
            # Text search
            schema = db_manager.get_table_schema(target_table)
            text_columns = [col['column_name'] for col in schema 
                        if col['data_type'] in ['character varying', 'text', 'varchar']]
            
            for column in text_columns:
                for term in search_terms:
                    where_conditions.append(f"{column} ILIKE %s")
                    params.append(f"%{term}%")
            
            # Add filters
            filter_conditions, filter_params = self.build_filter_conditions(filters, target_table)
            where_conditions.extend(filter_conditions)
            params.extend(filter_params)
            
            where_clause = " WHERE " + " OR ".join(where_conditions) if where_conditions else ""
            query = f"SELECT * FROM {target_table} {where_clause} LIMIT 10"
            
            result = db_manager.execute_query(query, params)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not result:
                return {
                    "answer": f"Tidak ditemukan data dalam tabel {self.get_table_display_name(target_table)}",
                    "data": [],
                    "intent": QueryIntent.SEARCH,
                    "table_used": target_table,
                    "sql_query": query,
                    "processing_time": processing_time
                }
            
            answer = f"Ditemukan {len(result)} hasil dalam tabel {self.get_table_display_name(target_table)}"
            
            return {
                "answer": answer,
                "data": result,
                "intent": QueryIntent.SEARCH,
                "table_used": target_table,
                "sql_query": query,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Search query failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": f"Maaf, pencarian gagal: {str(e)}",
                "data": [],
                "intent": QueryIntent.UNKNOWN,
                "table_used": target_table,
                "sql_query": None,
                "processing_time": processing_time
            }

    def execute_list_query(self, target_table: str, analysis: Dict) -> Dict[str, Any]:
        """Execute list query dengan semua required fields"""
        start_time = datetime.now()
        
        try:
            filters = analysis.get('filters', {})
            where_clause, params = self.build_where_clause(filters)
            query = f"SELECT * FROM {target_table} {where_clause} LIMIT 20"
            result = db_manager.execute_query(query, params)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not result:
                return {
                    "answer": f"Tidak ditemukan data dalam tabel {self.get_table_display_name(target_table)}",
                    "data": [],
                    "intent": QueryIntent.LIST,
                    "table_used": target_table,
                    "sql_query": query,
                    "processing_time": processing_time
                }
            
            answer = f"Data {self.get_table_display_name(target_table)}: {len(result)} records ditemukan"
            
            return {
                "answer": answer,
                "data": result,
                "intent": QueryIntent.LIST,
                "table_used": target_table,
                "sql_query": query,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"List query failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": f"Maaf, tidak dapat mengambil data: {str(e)}",
                "data": [],
                "intent": QueryIntent.UNKNOWN,
                "table_used": target_table,
                "sql_query": None,
                "processing_time": processing_time
            }

    def execute_aggregate_query(self, question: str, target_table: str, analysis: Dict) -> Dict[str, Any]:
        """Execute aggregate query dengan semua required fields"""
        start_time = datetime.now()
        
        try:
            # Simple aggregate for now
            if target_table == 'products':
                query = "SELECT AVG(price) as average_price FROM products"
                result = db_manager.execute_query(query)
                avg_price = result[0]['average_price'] if result else 0
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "answer": f"Rata-rata harga produk: {avg_price:,.2f}",
                    "data": [{"average_price": float(avg_price)}],
                    "intent": QueryIntent.AGGREGATE,
                    "table_used": target_table,
                    "sql_query": query,
                    "processing_time": processing_time
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": "Fitur agregasi untuk tabel ini belum tersedia",
                "data": [],
                "intent": QueryIntent.AGGREGATE,
                "table_used": target_table,
                "sql_query": None,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Aggregate query failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": f"Maaf, query agregasi gagal: {str(e)}",
                "data": [],
                "intent": QueryIntent.UNKNOWN,
                "table_used": target_table,
                "sql_query": None,
                "processing_time": processing_time
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

    # def get_table_display_name(self, table_name: str) -> str:
    #     """Get display name for table"""
    #     display_names = {
    #         'user_profiles': 'user',
    #         'products': 'produk', 
    #         'orders': 'order'
    #     }
    #     return display_names.get(table_name, table_name)

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