from __future__ import annotations

import os
import random
import numpy as np

def set_deterministic(seed: int = 224, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                raise RuntimeError(
                    "deterministic_torch=True but torch cannot enforce deterministic algorithms "
                    "for this environment/ops. Re-run without --deterministic_torch or use CPU."
                ) from e

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except Exception:
        pass
