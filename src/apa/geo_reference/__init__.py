"""
Georeferencing utilities for APA.

Provides coordinate conversion functions for UTM, ITM, and WGS84 coordinate systems.
"""

# This __init__.py makes geo_reference a Python package
# Modules are imported directly when needed to avoid dependency issues at import time
__all__ = ['CovertITM2LatLon', 'GetRoadsCoordinates', 'GetOptimalRoadOffset']

