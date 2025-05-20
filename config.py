class Config:
    def __init__(self):
        self.model_name = "google/flan-t5-large"
        self.embedding_model = "all-MiniLM-L6-v2"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_new_tokens = 300
        self.temperature = 0.3
        self.k_results = 2
        self.relevance_threshold = 0.7
        self.upload_folder = "uploads"
        self.index_folder = "indices"


config = Config()