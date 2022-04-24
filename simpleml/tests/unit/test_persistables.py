"""
Persistable related tests
"""

__author__ = "Elisha Yadgaran"

import unittest

from simpleml.persistables.base_persistable import Persistable

class PersistableTests(unittest.TestCase):
    def test_abstract_hash_error(self):
        """
        Confirm an error is raised if initialized without
        defining the hash
        """
        with self.assertRaises(TypeError):
            Persistable()

    def test_save_values(self):
        """
        should overwrite null fields on save and skip if provided
        """

    def test_load_from_dict(self):
        """
        should reinitialize object
        """

    def test_load_from_to_dict(self):
        """
        full roundtrip to clone
        """


if __name__ == "__main__":
    unittest.main(verbosity=2)
