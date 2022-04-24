"""
Persistence Serializers

These are the primary classes responsible for serializing the python objects
into storage formats

Convention is to serialize into the system temp folder and then pass off to
dedicated formatting (e.g type conversion, compression) and transport (local, cloud)
classes
"""

__author__ = "Elisha Yadgaran"
