"""Importing this package registers every shape builder."""

# Import order = order in the shape list the model sees: ready-made objects first.
from oprel.integrations.scene3d.engine.shapes import objects, primitives, profiles, paths, parts  # noqa: F401
