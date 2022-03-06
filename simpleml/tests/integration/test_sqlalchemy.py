"""
SqlAlchemy specific tests
"""

__author__ = "Elisha Yadgaran"


import unittest

from simpleml.orm.persistable import ORMPersistable


class MutableJSONTests(unittest.TestCase):
    """Default sqlalchemy behavior treats JSON data as immutable"""

    def test_modifying_json_field(self):
        """
        Top level JSON change
        '''
        persistable = ORMPersistable(name='top_level_json_modification_test')
        persistable.save()

        persistable.metadata_["new_key"] = "blah"
        self.assertIn(persistable, persistable._session.dirty)
        persistable._session.refresh(persistable)

    def test_modifying_nested_json_field(self):
        """
        Nested JSON change
        '''
        persistable = ORMPersistable(name='nested_json_modification_test')
        persistable.metadata_['new_key'] = {}
        persistable.save()

        persistable.metadata_["new_key"]["sub_key"] = "blah"
        self.assertIn(persistable, persistable._session.dirty)
        persistable._session.refresh(persistable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
