"""
init_db.py
==========
Creates all database tables defined in database.py using
Base.metadata.create_all(). Run this once before starting the ETL pipelines.

Usage:
    .venv\Scripts\python init_db.py
"""

from src.database.database import Base, engine

print("Creating all database tables...")
Base.metadata.create_all(bind=engine)
print("Done! All tables created successfully.")
