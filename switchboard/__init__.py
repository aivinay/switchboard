"""Switchboard package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("switchboard-local")
except PackageNotFoundError:
    __version__ = "0.2.2"
