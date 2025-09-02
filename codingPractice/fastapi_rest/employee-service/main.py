from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models, schemas, crud
from database import SessionLocal, engine, Base
from config.settings import settings
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Employee API")

@app.get("/env")
def read_env():
    return {
        "env": settings.env,
        "debug": settings.debug,
        "db_url": settings.database_url
    }

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/employees/", response_model=schemas.EmployeeOut)
def create_employee(emp: schemas.EmployeeCreate, db: Session = Depends(get_db)):
    return crud.create_employee(db, emp)

@app.get("/employees/", response_model=list[schemas.EmployeeOut])
def read_employees(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.get_employees(db, skip, limit)

@app.get("/employees/{emp_id}", response_model=schemas.EmployeeOut)
def read_employee(emp_id: int, db: Session = Depends(get_db)):
    emp = crud.get_employee(db, emp_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp

@app.put("/employees/{emp_id}", response_model=schemas.EmployeeOut)
def update_employee(emp_id: int, emp_data: schemas.EmployeeCreate, db: Session = Depends(get_db)):
    emp = crud.update_employee(db, emp_id, emp_data)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp

@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int, db: Session = Depends(get_db)):
    emp = crud.delete_employee(db, emp_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"detail": "Employee deleted"}
