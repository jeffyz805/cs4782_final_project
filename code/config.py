"""Configuration for Tree of Thoughts Code Debugger."""
import os

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("TOT_MODEL", "gpt-4o-mini")
MAX_TOKENS_HYPOTHESIS = 800
MAX_TOKENS_FIX = 600
MAX_TOKENS_EVAL = 50
TEMPERATURE = 0.7

# ToT search parameters
TOT_K = 3           # branching factor (candidates per step)
TOT_DEPTH = 2       # tree depth: 1=hypothesis, 2=fix
TOT_SEARCH = "bfs"  # "bfs" or "dfs"

# Evaluation mode: "llm", "execution", or "hybrid"
EVALUATOR = "hybrid"

# Execution
EXEC_TIMEOUT = 5    # seconds per test execution
MAX_RETRIES = 2     # LLM call retries on failure

# Experiment
NUM_PROBLEMS = 50   # number of problems to evaluate (set lower for demo)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Baselines
COT_SC_SAMPLES = 5  # self-consistency sample count

# Random seed for reproducibility
SEED = 42
