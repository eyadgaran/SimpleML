"""
ORM package

Focuses on all database related interaction, intentionally separated to allow
parallel Persistable objects to only deal with the glue interactions across
python types and libraries

Each mapped Persistable table model has a 1:1 parallel class
defined as a native python object
"""

__author__ = "Elisha Yadgaran"
