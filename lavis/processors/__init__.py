from lavis.processors.base_processor import BaseProcessor
from lavis.processors.blip_processors import (
    BlipImageBaseProcessor,
)
from lavis.common.registry import registry
__all__ = [
    "BaseProcessor",

    # BLIP
    "BlipImageBaseProcessor",
    
]