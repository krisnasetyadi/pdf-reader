# processor.py - Add these methods to PDFQAProcessor class

class PDFQAProcessor:
    def __init__(self):
        # Existing initialization...
        self.db_manager = None
        self._db_initialized = False

    def initialize_database(self):
        """Initialize database connection"""
        try:
            from database import db_manager
            self.db_manager = db_manager
            self._db_initialized = True
            logger.info("Database components initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            self._db_initialized = False

    def initialize_components(self):
        """Initialize all components including database"""
        with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing all components...")
            try:
                # Initialize embeddings and LLM (existing code)...
                # Your existing initialization code here...

                # Initialize database
                self.initialize_database()

                self._initialized = True
                logger.info("All components initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                raise

    def analyze_question_for_sql(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine if it's suitable for database query"""
        # Keywords that suggest database query
        db_keywords = [
            'user', 'profile', 'customer', 'product', 'order', 'price',
            'jumlah', 'total', 'data', 'tabel', 'table', 'database',
            'nama', 'email', 'alamat', 'tanggal', 'date'
        ]

        question_lower = question.lower()
        is_db_question = any(keyword in question_lower for keyword in db_keywords)

        # Try to identify target table
        target_table = None
        table_keywords = {
            'user': ['user', 'pengguna', 'customer', 'pelanggan'],
            'product': ['product', 'produk', 'barang'],
            'order': ['order', 'pesanan', 'transaksi']
        }

        for table, keywords in table_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                target_table = table
                break

        return {
            "is_db_question": is_db_question,
            "target_table": target_table,
            "search_terms": self.extract_search_terms(question)
        }

    def extract_search_terms(self, question: str) -> List[str]:
        """Extract meaningful search terms from question"""
        # Remove common stop words
        stop_words = {'apa', 'siapa', 'dimana', 'kapan', 'berapa', 'bagaimana', 'yang', 'dan', 'atau', 'di', 'ke', 'dari'}
        words = question.lower().split()
        meaningful_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return meaningful_terms

    def query_database(self, question: str, table_name: str = None) -> Dict[str, Any]:
        """Query database based on natural language question"""
        if not self._db_initialized:
            return {"error": "Database not initialized"}

        analysis = self.analyze_question_for_sql(question)

        if not analysis["is_db_question"]:
            return {"error": "Question not suitable for database query"}

        # Determine which tables to search
        tables_to_search = []
        if table_name:
            tables_to_search = [table_name]
        elif analysis["target_table"]:
            tables_to_search = [analysis["target_table"]]
        else:
            # Search in all configured tables
            tables_to_search = config.db_tables

        results = []
        for table in tables_to_search:
            try:
                table_results = self.db_manager.search_in_table(
                    table, analysis["search_terms"], limit=5
                )
                for result in table_results:
                    result['_table'] = table
                results.extend(table_results)
            except Exception as e:
                logger.error(f"Search in table {table} failed: {str(e)}")
                continue

        return {
            "results": results,
            "table_used": tables_to_search[0] if tables_to_search else None,
            "search_terms": analysis["search_terms"]
        }

    def generate_sql_based_answer(self, db_results: Dict[str, Any], question: str) -> str:
        """Generate natural language answer from database results"""
        if "error" in db_results:
            return db_results["error"]

        results = db_results.get("results", [])
        if not results:
            return "Tidak ditemukan data yang sesuai dalam database."

        table_name = db_results.get("table_used", "database")

        if len(results) == 1:
            result = results[0]
            answer = f"Ditemukan data dalam tabel {table_name}:\n"
            for key, value in result.items():
                if not key.startswith('_'):  # Skip internal fields
                    answer += f"- {key}: {value}\n"
        else:
            answer = f"Ditemukan {len(results)} hasil dalam tabel {table_name}:\n"
            for i, result in enumerate(results[:3], 1):  # Show first 3 results
                answer += f"\nHasil {i}:\n"
                for key, value in list(result.items())[:4]:  # Show first 4 fields
                    if not key.startswith('_'):
                        answer += f"  - {key}: {value}\n"

        return answer
