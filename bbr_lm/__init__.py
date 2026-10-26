"""Modular language-model framework for BBR pacing-gain prediction."""

import sys
import types

from bbr_lm.data import exp_pool as _exp_pool

# Register the legacy module path used by pickled experience pools
# (see bbr_lm/data/exp_pool.py).
if "utils.exp_pool" not in sys.modules:
    _legacy_package = sys.modules.setdefault("utils", types.ModuleType("utils"))
    _legacy_package.exp_pool = _exp_pool
    sys.modules["utils.exp_pool"] = _exp_pool
