"""
Configuration Module for Disaster Literacy RAG System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== BASE PATHS ====================
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== KB SETTINGS ====================
KB_DOCUMENTS_DIR = DATA_DIR / "kb_documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
METADATA_FILE = DATA_DIR / "kb_metadata.json"

CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", 75))
MIN_CHUNK_SIZE = 100

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler\Library\bin")
OCR_LANGUAGE = "eng"
DPI = 300

# ==================== EMBEDDINGS ====================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# ==================== VECTOR STORE ====================
VECTOR_DB_TYPE = "faiss"
FAISS_INDEX_TYPE = "Flat"
VECTOR_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
CHUNK_METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.pkl"

# ==================== RETRIEVAL ====================
# Legacy offline retrieval (defaults to economy)
OFFLINE_TOP_K_RETRIEVAL = int(os.getenv("OFFLINE_TOP_K_RETRIEVAL", 2))

# Economy mode - retrieve fewer chunks for lower spec systems
OFFLINE_ECONOMY_TOP_K_RETRIEVAL = int(os.getenv("OFFLINE_ECONOMY_TOP_K_RETRIEVAL", 2))

# Power mode - retrieve more chunks for better context
OFFLINE_POWER_TOP_K_RETRIEVAL = int(os.getenv("OFFLINE_POWER_TOP_K_RETRIEVAL", 6))

# Online retrieval
ONLINE_TOP_K_RETRIEVAL = int(os.getenv("ONLINE_TOP_K_RETRIEVAL", 10))
RETRIEVAL_TYPE = "dense"

# ==================== LLM SETTINGS ====================
# Economy mode - fewer tokens for lower spec systems (fewer chunks = less context)
OFFLINE_ECONOMY_MAX_LLM_TOKENS = int(os.getenv("OFFLINE_ECONOMY_MAX_LLM_TOKENS", 512))

# Power mode - more tokens to handle larger context from more chunks
OFFLINE_POWER_MAX_LLM_TOKENS = int(os.getenv("OFFLINE_POWER_MAX_LLM_TOKENS", 1024))

# Online tokens
ONLINE_MAX_LLM_TOKENS = int(os.getenv("ONLINE_MAX_LLM_TOKENS", 2048))
TEMPERATURE = 0.3
LLM_TIMEOUT = 60

# === Offline Model ===
# Economy mode - for lower spec systems (current default)
OFFLINE_LLM_MODEL_PATH_ECONOMY = os.getenv(
    "OFFLINE_LLM_MODEL_PATH_ECONOMY",
    str(MODELS_DIR / "llama-2-7b-chat.Q4_K_M.gguf")
)

# Power mode - for higher spec systems (more capable model)
OFFLINE_LLM_MODEL_PATH_POWER = os.getenv(
    "OFFLINE_LLM_MODEL_PATH_POWER",
    str(MODELS_DIR / "qwen2-7b-instruct-q4_k_m.gguf")
)

OFFLINE_LLM_CONTEXT_LENGTH = 4096
OFFLINE_LLM_THREADS = 4

# Default offline mode
DEFAULT_OFFLINE_MODE = "economy"

# === Online Provider Defaults ===
# Default provider = Google (user can choose OpenRouter via UI)
ONLINE_LLM_PROVIDER = os.getenv("ONLINE_LLM_PROVIDER", "google")

# Translation settings (only applies when using Google provider)
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "false").lower() == "true"

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = "gemini-2.0-flash"  # Better free tier limits: 15 RPM vs 10 RPM

# OpenRouter ✅
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct:free")

# ==================== MODES ====================
SUPPORTED_MODES = ["Advisory", "Educational", "Simulation"]
DEFAULT_MODE = "Advisory"

MIN_MCQ_COUNT = 10
MCQ_OPTIONS = 4
SIMULATION_SCENARIO_LENGTH = "2-3 paragraphs"

# ==================== LOGGING ====================
ENABLE_FALLBACK = True
FALLBACK_MESSAGE = "Unable to generate full guidance. Please follow the essential safety actions below."
ERROR_LOG_FILE = LOGS_DIR / "errors.log"
GENERAL_LOG_FILE = LOGS_DIR / "system.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ==================== SAFETY ====================
SAFETY_DISCLAIMER = (
    "⚠️ This tool gives informational disaster guidance. "
    "In emergencies, follow local authorities."
)

GROUNDING_INSTRUCTION = (
    "Provide answers only using the CONTEXT. "
    "If unknown say 'I don't know – contact authorities'. "
    "Cite sources like [doc1] for every fact."
)

# ==================== SYSTEM ====================
HARDWARE_PROFILE = {"ram_gb": 8, "cpu_cores": 4, "gpu": None, "expected_latency_sec": 20}
ENABLE_VERSION_TRACKING = True
VERSION_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
ENABLE_METRICS = True
