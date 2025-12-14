# join_processor.py
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from database import db_manager

logger = logging.getLogger(__name__)

class JoinQueryProcessor:
    def __init__(self, structured_processor):
        self.structured_processor = structured_processor
        self.db_manager = db_manager
        
        # Common join patterns
        self.join_patterns = {
            'user_order': [
                r'user.*order|order.*user',
                r'pelanggan.*pesanan|pesanan.*pelanggan',
                r'customer.*transaction|transaction.*customer'
            ],
            'product_order': [
                r'produk.*order|order.*produk',
                r'product.*transaction|transaction.*product',
                r'barang.*pesanan|pesanan.*barang'
            ],
            'user_product': [
                r'user.*produk|produk.*user',
                r'pelanggan.*barang|barang.*pelanggan'
            ]
        }
        
        # Auto-discovered relationships cache
        self.table_relationships = {}
        self.discover_relationships()

    def discover_relationships(self):
        """Discover potential table relationships from schema"""
        try:
            logger.info("🔍 Discovering table relationships...")
            
            tables = list(self.structured_processor.table_schemas.keys())
            
            for i, table1 in enumerate(tables):
                for table2 in tables[i+1:]:
                    # Check for foreign key relationships
                    relationship = self.detect_relationship(table1, table2)
                    if relationship:
                        key = f"{table1}_{table2}"
                        self.table_relationships[key] = relationship
                        logger.info(f"  Found relationship: {table1} ↔ {table2}")
            
            logger.info(f"✅ Discovered {len(self.table_relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Relationship discovery failed: {str(e)}")

    def detect_relationship(self, table1: str, table2: str) -> Optional[Dict[str, Any]]:
        """Detect relationship between two tables"""
        try:
            # Get schemas
            schema1 = db_manager.get_table_schema(table1)
            schema2 = db_manager.get_table_schema(table2)
            
            # Look for foreign key patterns
            for col1 in schema1:
                col1_name = col1['column_name'].lower()
                
                # Check if column references other table's id
                if col1_name.endswith('_id') or col1_name == f"{table2}_id":
                    # Found potential foreign key
                    return {
                        'table1': table1,
                        'table2': table2,
                        'join_condition': f"{table1}.{col1['column_name']} = {table2}.id",
                        'relationship_type': 'many-to-one',
                        'direction': f"{table1} → {table2}"
                    }
            
            for col2 in schema2:
                col2_name = col2['column_name'].lower()
                
                if col2_name.endswith('_id') or col2_name == f"{table1}_id":
                    return {
                        'table1': table1,
                        'table2': table2,
                        'join_condition': f"{table1}.id = {table2}.{col2['column_name']}",
                        'relationship_type': 'one-to-many',
                        'direction': f"{table1} ← {table2}"
                    }
            
            # Check for common column names (simple heuristic)
            col_names1 = {col['column_name'].lower() for col in schema1}
            col_names2 = {col['column_name'].lower() for col in schema2}
            common_cols = col_names1.intersection(col_names2)
            
            # Remove common generic columns
            generic_cols = {'id', 'created_at', 'updated_at', 'status'}
            meaningful_common = common_cols - generic_cols
            
            if meaningful_common:
                join_col = list(meaningful_common)[0]
                return {
                    'table1': table1,
                    'table2': table2,
                    'join_condition': f"{table1}.{join_col} = {table2}.{join_col}",
                    'relationship_type': 'common-column',
                    'direction': 'bidirectional'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect relationship: {str(e)}")
            return None

    def analyze_join_query(self, question: str) -> Dict[str, Any]:
        """Analyze question for join requirements"""
        question_lower = question.lower()
        
        # Detect which tables are mentioned
        mentioned_tables = []
        for table in self.structured_processor.table_schemas.keys():
            table_display = self.structured_processor.get_table_display_name(table)
            if (table.lower() in question_lower or 
                table_display in question_lower or
                any(word in question_lower for word in table_display.split())):
                mentioned_tables.append(table)
        
        # If only one table mentioned, check if join is implied
        if len(mentioned_tables) < 2:
            # Look for join keywords
            join_keywords = ['dan', 'dengan', 'beserta', 'serta', 'termasuk', 'plus', '+', '&']
            if any(keyword in question_lower for keyword in join_keywords):
                # Try to infer second table from context
                if mentioned_tables:
                    primary_table = mentioned_tables[0]
                    # Find related tables
                    related = self.find_related_tables(primary_table)
                    if related:
                        mentioned_tables.append(related[0])
        
        # Find best join relationship
        join_details = None
        if len(mentioned_tables) >= 2:
            join_details = self.find_best_join(mentioned_tables[:2])
        
        # Extract filters
        filters = self.extract_join_filters(question, mentioned_tables)
        
        return {
            'mentioned_tables': mentioned_tables,
            'join_details': join_details,
            'filters': filters,
            'requires_join': join_details is not None,
            'join_type': self.determine_join_type(question)
        }

    def find_related_tables(self, table: str) -> List[str]:
        """Find tables related to the given table"""
        related = []
        for rel_key, rel in self.table_relationships.items():
            if table in [rel['table1'], rel['table2']]:
                other = rel['table2'] if rel['table1'] == table else rel['table1']
                related.append(other)
        return related

    def find_best_join(self, tables: List[str]) -> Optional[Dict[str, Any]]:
        """Find the best join relationship between tables"""
        # Try direct relationship
        for rel_key, rel in self.table_relationships.items():
            if set([rel['table1'], rel['table2']]) == set(tables):
                return rel
        
        # Try reverse relationship
        for rel_key, rel in self.table_relationships.items():
            if set([rel['table1'], rel['table2']]) == set(tables[::-1]):
                # Return reversed relationship
                return {
                    'table1': rel['table2'],
                    'table2': rel['table1'],
                    'join_condition': rel['join_condition'],
                    'relationship_type': rel['relationship_type'],
                    'direction': 'reversed'
                }
        
        return None

    def extract_join_filters(self, question: str, tables: List[str]) -> Dict[str, List[Dict]]:
        """Extract filters for join query"""
        filters = {table: [] for table in tables}
        question_lower = question.lower()
        
        # Simple filter extraction - can be enhanced
        for table in tables:
            table_display = self.structured_processor.get_table_display_name(table)
            
            # Look for table-specific filters
            if table_display in question_lower:
                # Extract filters after table mention
                # This is simplified - can be enhanced with NLP
                pass
        
        return filters

    def determine_join_type(self, question: str) -> str:
        """Determine join type from question"""
        question_lower = question.lower()
        
        if 'beserta' in question_lower or 'termasuk' in question_lower:
            return 'LEFT JOIN'
        elif 'hanya' in question_lower or 'yang ada' in question_lower:
            return 'INNER JOIN'
        else:
            return 'INNER JOIN'  # default

    def execute_join_query(self, question: str, join_type: str = "auto") -> Dict[str, Any]:
        """Execute join query based on natural language"""
        start_time = datetime.now()
        
        try:
            # Analyze query
            analysis = self.analyze_join_query(question)
            
            if not analysis['requires_join'] or not analysis['join_details']:
                # Fallback to single table query
                return self.fallback_to_single_table(question, analysis)
            
            join_details = analysis['join_details']
            table1 = join_details['table1']
            table2 = join_details['table2']
            
            # Build join query
            if join_type == "auto":
                join_type = analysis['join_type']
            
            # Select columns with table prefixes to avoid ambiguity
            cols1 = self.get_select_columns(table1, prefix=table1)
            cols2 = self.get_select_columns(table2, prefix=table2)
            select_cols = cols1 + cols2
            
            # Build WHERE clause from filters
            where_conditions = []
            params = []
            
            for table, table_filters in analysis['filters'].items():
                for filt in table_filters:
                    where_conditions.append(f"{table}.{filt['column']} {filt['operator']} %s")
                    params.append(filt['value'])
            
            where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Construct SQL
            sql_query = f"""
                SELECT {', '.join(select_cols)}
                FROM {table1}
                {join_type} {table2} ON {join_details['join_condition']}
                {where_clause}
                LIMIT 20
            """
            
            # Execute query
            results = db_manager.execute_query(sql_query, tuple(params))
            
            # Generate natural language answer
            answer = self.generate_join_answer(
                results, table1, table2, join_details, analysis
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": answer,
                "data": results,
                "tables_used": [table1, table2],
                "join_conditions": [{
                    "table1": table1,
                    "table2": table2,
                    "condition": join_details['join_condition'],
                    "type": join_type
                }],
                "sql_query": sql_query.strip(),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Join query failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": f"Maaf, query join gagal: {str(e)}",
                "data": [],
                "tables_used": [],
                "join_conditions": [],
                "sql_query": None,
                "processing_time": processing_time
            }

    def get_select_columns(self, table: str, prefix: str = None) -> List[str]:
        """Get select columns for a table"""
        schema = db_manager.get_table_schema(table)
        columns = []
        
        for col in schema:
            col_name = col['column_name']
            if prefix:
                columns.append(f"{prefix}.{col_name} as {prefix}_{col_name}")
            else:
                columns.append(col_name)
        
        return columns[:8]  # Limit columns

    def generate_join_answer(self, results: List[Dict], table1: str, table2: str, 
                            join_details: Dict, analysis: Dict) -> str:
        """Generate natural language answer for join results"""
        if not results:
            return f"Tidak ditemukan data dari join {table1} dan {table2}"
        
        display1 = self.structured_processor.get_table_display_name(table1)
        display2 = self.structured_processor.get_table_display_name(table2)
        
        answer = f"Ditemukan {len(results)} hasil dari join {display1} dan {display2}:\n\n"
        
        # Show first few results
        for i, row in enumerate(results[:3]):
            answer += f"Result {i+1}:\n"
            
            # Extract and format key values
            key_values = []
            for key, value in row.items():
                if key.endswith('_name') or key.endswith('_email') or '_id' in key:
                    key_values.append(f"{key}: {value}")
            
            if key_values:
                answer += "  " + ", ".join(key_values[:4]) + "\n"
        
        if len(results) > 3:
            answer += f"\n... dan {len(results) - 3} hasil lainnya."
        
        return answer

    def fallback_to_single_table(self, question: str, analysis: Dict) -> Dict[str, Any]:
        """Fallback to single table query when join not possible"""
        from models import QueryIntent
        
        # Use structured processor for single table
        result = self.structured_processor.execute_structured_query(question)
        
        # Convert to join response format
        return {
            "answer": result['answer'] + "\n\n(Catatan: Query dieksekusi pada single table, bukan join)",
            "data": result['data'],
            "tables_used": [result['table_used']] if result['table_used'] != 'system' else [],
            "join_conditions": [],
            "sql_query": result.get('sql_query'),
            "processing_time": result['processing_time']
        }

# Global instance - will be initialized after structured_processor
join_processor = None

def initialize_join_processor(structured_proc):
    global join_processor
    join_processor = JoinQueryProcessor(structured_proc)
    return join_processor