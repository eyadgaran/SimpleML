'''
Platform independent sqlalchemy types
'''

from sqlalchemy.types import TypeDecorator, CHAR, JSON as SQLJSON, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy_json import mutable_json_type
import uuid


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    http://docs.sqlalchemy.org/en/latest/core/custom_types.html
    """
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                # hexstring
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


class JSON(TypeDecorator):
    '''
    Platform-independent JSON type

    Uses PostgreSQL's JSONB type, otherwise falls back to standard JSON
    '''
    impl = SQLJSON

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB(astext_type=Text()))
        else:
            return dialect.type_descriptor(SQLJSON())


# Mutable version of JSON field
# https://docs.sqlalchemy.org/en/13/core/type_basics.html?highlight=json#sqlalchemy.types.JSON
MutableJSON = mutable_json_type(dbtype=JSON, nested=True)
