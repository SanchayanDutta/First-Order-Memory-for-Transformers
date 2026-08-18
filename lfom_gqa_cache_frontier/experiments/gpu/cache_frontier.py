# Thin wrapper for backward compatibility. Run ../cache_frontier.py instead.
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'cache_frontier.py'), run_name='__main__')
