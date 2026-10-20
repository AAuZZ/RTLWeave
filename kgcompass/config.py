from dotenv import load_dotenv
import json
import os
from github import Github

load_dotenv()

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# --- model config -----------------------------------------------------------
# Profiles stored in model_configs.json.  Switch via MODEL_PROFILE in .env.
_MODEL_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "model_configs.json")
_model_cfg: dict = {}
try:
    with open(os.path.abspath(_MODEL_CFG_PATH)) as f:
        _model_cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pass

_model_profile = os.getenv("MODEL_PROFILE", "ds-v4-flash")
_model = _model_cfg.get(_model_profile, {})

BAILIAN_API_KEY = _model.get("api_key") or os.getenv("BAILIAN_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _model.get("base_url") or os.getenv(
    "OPENAI_COMPAT_BASE_URL",
    os.getenv("DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL_NAME = _model.get("model_name") or os.getenv("MODEL_NAME", "deepseek-v4-flash")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

# Knowledge Graph Configuration
MAX_CANDIDATE_METHODS = 500
MAX_SEARCH_DEPTH = 2
SEARCH_SPACE = 50

# Connection strength coefficients
CONNECTION_FACTOR = 0.5
WEAK_CONNECTION = 1.0
NORMAL_CONNECTION = WEAK_CONNECTION * CONNECTION_FACTOR
STRONG_CONNECTION = NORMAL_CONNECTION * CONNECTION_FACTOR

# Dataset Configuration
DATASET_NAME = os.getenv("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
LOCAL_BENCHMARK_PATH = os.getenv("LOCAL_BENCHMARK_PATH")

# Graph Configuration
DECAY_FACTOR = 0.6
VECTOR_SIMILARITY_WEIGHT = 0.6

# LLM Request Configuration
MAX_TOKENS = 8192
TEMPERATURE = 0.3
TOP_P = 0.95

MAX_INPUT_LENGTH = 1000000  # 字符，~300K tokens（模型支持 1M）
LLM_LOC_MAX = 10
CANDIDATE_LOCATIONS_MAX = 20
