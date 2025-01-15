# handling all the helper functions which are called multiple times in a project
# - Logging functions

from .logs import MakeLogger

# To import all modules/functions at once use the *
__all__ = (
    MakeLogger
)