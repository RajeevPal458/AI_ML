from db.database import Base,engine
from models.employee import Employee

Base.metadata.create_all(bind=engine)

### use command to crete all tables => python -m  db.create_tables 