'''
Base class for sqlalchemy
'''

__author__ = 'Elisha Yadgaran'


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, func
from sqlalchemy_mixins import AllFeaturesMixin
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects import postgresql


Base = declarative_base()


class BaseSQLAlchemy(Base, AllFeaturesMixin):
    '''
    Base class for all SimpleML database objects. Defaults to PostgreSQL
    but can be swapped out for any supported SQLAlchemy backend.

    Takes advantage of sqlalchemy-mixins to enable active record operations
    (TableModel.save(), create(), where(), destroy())

    Added some inheritable convenience methods

    -------
    Schema
    -------
    created_timestamp: Server time on insert
    modified_timestamp: Server time on update
    '''
    __abstract__ = True

    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())

    @staticmethod
    def compile_query(query):
        compiler = query.compile if not hasattr(query, 'statement') else query.statement.compile
        return compiler(dialect=postgresql.dialect())

    @classmethod
    def upsert(cls, df, no_update_cols=[], created_column='created_timestamp'):
        '''
        Upsert statement requires PostgreSQL >= 9.5
        '''
        instantiated_class = cls._instantiate()
        table = instantiated_class.__table__

        # Force no update on created timestamp column
        no_update_cols.append(created_column)

        stmt = insert(table).values(df.to_dict('records'))

        update_cols = [c.name for c in table.c
                       if c not in list(table.primary_key.columns)
                       and c.name not in no_update_cols]

        on_conflict_stmt = stmt.on_conflict_do_update(
            index_elements=table.primary_key.columns,
            set_={k: getattr(stmt.excluded, k) for k in update_cols}
        )

        instantiated_class._session.execute(on_conflict_stmt)

    @classmethod
    def filter(cls, *filters):
        instantiated_class = cls._instantiate()
        return instantiated_class._session.query(cls).filter(*filters)

    @classmethod
    def query_by(cls, *queries):
        instantiated_class = cls._instantiate()
        return instantiated_class._session.query(*queries)

    @classmethod
    def _instantiate(cls):
        return cls()
