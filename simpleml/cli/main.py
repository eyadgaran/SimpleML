"""
Main entrypoint for the `simpleml` command line
"""

__author__ = "Elisha Yadgaran"

import click

from . import database


@click.group()
def cli():
    """
    Entrypoint for the `simpleml` command line call
    """


cli.add_command(database.db)
