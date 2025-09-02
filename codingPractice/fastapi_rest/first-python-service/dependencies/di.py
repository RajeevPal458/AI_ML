from db.database import LocalSession
from fastapi import Depends

def get_db():
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()