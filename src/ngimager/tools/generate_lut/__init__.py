"""
Light-response LUT generation tools for NOVO scintillators.

This subpackage exposes:

- :mod:`ngimager.tools.generate_lut.NOVO_light_response_functions` —
  the script and helpers used to build E(L) LUTs for M600 and OGS
  from SRIM stopping power data and calibration measurements.
"""

from . import NOVO_light_response_functions

__all__ = ["NOVO_light_response_functions"]
