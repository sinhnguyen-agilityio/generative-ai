from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vfs_root: str = ".data/vfs"
    checkpoint_path: str = ".data/checkpoints.json"
    vector_path: str = ".data/vector-index.json"
    output_dir: str = "exports"
    max_search_calls: int = 5
    recovery_attempts: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            vfs_root=os.getenv("INTELLIGENCE_VFS_ROOT", cls.vfs_root),
            checkpoint_path=os.getenv("INTELLIGENCE_CHECKPOINT_PATH", cls.checkpoint_path),
            vector_path=os.getenv("INTELLIGENCE_VECTOR_PATH", cls.vector_path),
            output_dir=os.getenv("INTELLIGENCE_OUTPUT_DIR", cls.output_dir),
            max_search_calls=min(int(os.getenv("INTELLIGENCE_MAX_SEARCH_CALLS", "5")), 5),
            recovery_attempts=max(1, int(os.getenv("INTELLIGENCE_RECOVERY_ATTEMPTS", "3"))),
        )

