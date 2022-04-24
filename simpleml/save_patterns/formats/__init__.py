"""
Persistence Formats

These are the in-between classes responsible for type conversion and formatting

Convention is to use a serializer to serialize python objects into the system
temp folder first and then call a formatter to manipulate the tempfiles
(e.g. compression) before handing off to the transport class for durable storage
"""

__author__ = "Elisha Yadgaran"
