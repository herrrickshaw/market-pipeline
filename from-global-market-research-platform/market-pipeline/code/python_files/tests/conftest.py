"""Shared test setup: offline, side-effect-free imports of the pipeline modules.

Every module under test lives one directory up and resolves its data paths at
import time (data_registry.MARKET_CACHE, obs.LOGDIR). Point those at a throwaway
directory BEFORE anything is imported so no test can read or write the real
cache, ledger, warehouse or log tree. env_loader is replaced with a stub so no
test can ever open a credentials file.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="mp_tests_"))
os.environ["MARKET_CACHE"] = str(_SANDBOX / "market_cache")
os.environ["MARKET_LOGDIR"] = str(_SANDBOX / "logs")
os.environ["MARKET_WH"] = str(_SANDBOX / "warehouse")

CODE = Path(__file__).resolve().parent.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

_stub = types.ModuleType("env_loader")
_stub.get = lambda key, default="": default
_stub.require = lambda *keys: {}
_stub.present = lambda *keys: False
sys.modules["env_loader"] = _stub
