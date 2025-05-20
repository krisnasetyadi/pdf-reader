from config import config
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
import logging
from langchain.llms import HuggingFacePipeline
import torch

logger = logging.getLogger(__name__)


class PDFQAProcessor:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.initialize_components()

    def initialize_components(self):
        """Initialize ML components"""
        logger.info("Initializing NLP components...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() 
                else torch.float32
            )
            generation_config = GenerationConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=True,
                top_p=0.95
            )
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=self.tokenizer,
                generation_config=generation_config
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise


processor = PDFQAProcessor()