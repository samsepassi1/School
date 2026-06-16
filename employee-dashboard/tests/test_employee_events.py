import pytest
from pathlib import Path

# project_root = 2 levels up from this file (tests/ → project root)
project_root = Path(__file__).resolve().parent.parent

# apply the pytest fixture decorator to a `db_path` function


@pytest.fixture
def db_path():

    # return a pathlib object for the `employee_events.db` file
    return project_root / "python-package" / \
        "employee_events" / "employee_events.db"

# test that the database file exists


def test_db_exists(db_path):

    # assert that the sqlite database file exists
    assert db_path.is_file()


@pytest.fixture
def db_conn(db_path):
    from sqlite3 import connect
    return connect(db_path)


@pytest.fixture
def table_names(db_conn):
    name_tuples = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    return [x[0] for x in name_tuples]

# test that the 'employee' table exists


def test_employee_table_exists(table_names):

    assert 'employee' in table_names

# test that the 'team' table exists


def test_team_table_exists(table_names):

    assert 'team' in table_names

# test that the 'employee_events' table exists


def test_employee_events_table_exists(table_names):

    assert 'employee_events' in table_names
