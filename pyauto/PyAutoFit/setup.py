import os
from setuptools import setup

# Source builds stamp a version ABOVE every date release. Intra-family deps
# carry `>=` floors, so a low dev stamp (the old "1.0.dev0") makes a local
# checkout unsatisfiable against its own siblings. Release builds always set
# VERSION explicitly, so this default is never published.
version = os.environ.get("VERSION", "2026.10.23.1.post1")

setup(
    version=version,
)