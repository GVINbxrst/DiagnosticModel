#!/usr/bin/env python
"""Создание базы данных PostgreSQL (diagmod) при наличии прав superuser/createdb.

Использование (переменные окружения или аргументы):
  python scripts/create_database.py --db diagmod --user postgres --password postgres --host localhost --port 5432

Если БД уже существует — завершает работу без ошибки.
"""
from __future__ import annotations
import argparse
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def parse_args():
    ap = argparse.ArgumentParser(description="Create PostgreSQL database if absent")
    ap.add_argument('--db', default=os.environ.get('PGDATABASE', 'diagmod'))
    ap.add_argument('--user', default=os.environ.get('PGUSER', 'postgres'))
    ap.add_argument('--password', default=os.environ.get('PGPASSWORD', 'postgres'))
    ap.add_argument('--host', default=os.environ.get('PGHOST', 'localhost'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('PGPORT', '5432')))
    return ap.parse_args()


def database_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        return cur.fetchone() is not None


def create_db_if_needed(args):
    # Подключаемся к стандартной базе postgres
    try:
        conn = psycopg2.connect(dbname='postgres', user=args.user, password=args.password, host=args.host, port=args.port)
    except Exception as e:
        print(f"ERROR: Не удалось подключиться к серверу Postgres: {e}", file=sys.stderr)
        sys.exit(1)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        if database_exists(conn, args.db):
            print(f"БД '{args.db}' уже существует - пропуск")
            return 0
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE {args.db} WITH ENCODING 'UTF8' TEMPLATE template1")
        print(f"Создана база данных '{args.db}'")
        return 0
    except Exception as e:
        print(f"ERROR: Не удалось создать БД: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(create_db_if_needed(parse_args()))
