__version__ = "0.0.1"

# Import subpackages
from . import planners
from . import mapping
from . import utility

# Import all classes from subpackages (respects their __all__ definitions)
from .mapping import *
from .planners import *
from .utility import *

# Automatically combine __all__ from subpackages
# This way, changes in subpackages automatically propagate here
__all__ = [
    # Subpackages
    "planners",
    "mapping",
    "utility",
] + mapping.__all__ + planners.__all__ + utility.__all__