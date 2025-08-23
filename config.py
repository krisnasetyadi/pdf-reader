# config.py
class Config:
    def __init__(self):
        self.model_name = "google/flan-t5-large"
        self.embedding_model = "all-MiniLM-L6-v2"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_new_tokens = 300
        self.temperature = 0.3
        self.k_results = 2
        self.k_per_collection = 3  # New: Docs per collection to retrieve
        self.total_k_results = 10  # New: Total docs to consider
        self.relevance_threshold = 0.7
        self.upload_folder = "uploads"
        self.index_folder = "indices"


config = Config()

# class Config:
#     def __init__(self):
#         self.model_name = "google/flan-t5-large"
#         self.embedding_model = "all-MiniLM-L6-v2"
#         self.chunk_size = 1000
#         self.chunk_overlap = 200
#         self.max_new_tokens = 300
#         self.temperature = 0.3
#         self.k_results = 2
#         self.relevance_threshold = 0.7
#         self.upload_folder = "uploads"
#         self.index_folder = "indices"


# config = Config()