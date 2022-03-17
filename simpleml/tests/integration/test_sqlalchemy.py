"""
SqlAlchemy specific tests
"""

__author__ = "Elisha Yadgaran"


import unittest
import uuid

from simpleml.orm.metadata import SimplemlCoreSqlalchemy
from simpleml.orm.sqlalchemy_types import GUID, MutableJSON
from sqlalchemy import Column


class MutableJSONTests(unittest.TestCase):
    '''Default sqlalchemy behavior treats JSON data as immutable'''

    class JSONTestClass(ORMPersistable):
        __tablename__ = 'json_tests'

    @classmethod
    def setUpClass(cls):
        cls.JSONTestClass.__table__.create(checkfirst=True)

    def test_modifying_json_field(self):
        """
        Top level JSON change
        """
        persistable = self.JSONTestClass()
        persistable.json_col = {}
        persistable.save()

        persistable.json_col["new_key"] = "blah"
        self.assertIn(persistable, persistable._session.dirty)
        persistable._session.refresh(persistable)

    def test_modifying_nested_json_field(self):
        """
        Nested JSON change
        """
        persistable = self.JSONTestClass()
        persistable.json_col = {}
        persistable.json_col["new_key"] = {}
        persistable.save()

        persistable.json_col["new_key"]["sub_key"] = "blah"
        self.assertIn(persistable, persistable._session.dirty)
        persistable._session.refresh(persistable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
