"""
Executor frameworks

An executor is the backend that operates over persistables to use and create
artifacts. A central characteristic of executors is that they are interchangeable;
outputs and artifacts will be the same. There is NO GUARANTEE that all persistable
types will be compatible with all executor frameworks. There is also no guarantee
that executors support native operations for the persistable types. Outputs will
be the same, but using different executors may result in more efficient processing
"""

__author__ = "Elisha Yadgaran"
