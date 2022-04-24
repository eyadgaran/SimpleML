"""
Adapters for compatibility between persistables and executors. Adapters are the
pluggable wrappers for persistables that ensure interoperability between
steps in processing.

Adapters must not contain any logic central to the processing. They can only be
interfaces to align formats, signatures, etc for consistent behavior.
"""

__author__ = "Elisha Yadgaran"
