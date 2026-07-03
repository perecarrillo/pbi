"""
This package contains the modular pipeline components. Import the orchestrator to run the full pipeline, or import
individual modules for fine-grained control.
"""

from pipeline.orchestrator import run_full_pipeline, run_embed_only

# Only export run_full_pipeline and run_embed_only
__all__ = [
    "run_full_pipeline",
    "run_embed_only",
]
