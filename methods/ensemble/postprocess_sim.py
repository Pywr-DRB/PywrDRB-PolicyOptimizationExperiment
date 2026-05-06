#!/usr/bin/env python3
"""
Post-optimization batch simulations: independent reservoir model + Pywr-DRB ensemble / selected picks.

This is a thin entrypoint for ``methods.ensemble.run_pareto_simulations`` (no matplotlib).

Examples::

  python -m methods.ensemble.postprocess_sim simulate --mode indie
  python -m methods.ensemble.postprocess_sim simulate --mode selected
  python -m methods.ensemble.postprocess_sim simulate --mode ensemble --output outputs/pareto_ensemble_pywr.pkl
  python -m methods.ensemble.postprocess_sim simulate --mode all
  python -m methods.ensemble.postprocess_sim figure12 --bundle outputs/pareto_ensemble_pywr.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ensemble.run_pareto_simulations import main

if __name__ == "__main__":
    main()
