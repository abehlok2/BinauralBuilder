"""Headless foundations for the SAM/HRTF spatial-audio workbench.

This package deliberately has no GUI imports.  It is the stable boundary used
by future renderers, command-line tools, and the PyQt presentation layer.
"""

from .model import Project, ProjectValidationError, load_project, save_project

__all__ = ["Project", "ProjectValidationError", "load_project", "save_project"]
