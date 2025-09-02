from sqlalchemy.orm import Session
import models, schemas

def create_employee(db: Session, emp: schemas.EmployeeCreate):
    db_emp = models.Employee(**emp.dict())
    db.add(db_emp)
    db.commit()
    db.refresh(db_emp)
    return db_emp

def get_employees(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Employee).offset(skip).limit(limit).all()

def get_employee(db: Session, emp_id: int):
    return db.query(models.Employee).filter(models.Employee.id == emp_id).first()

def update_employee(db: Session, emp_id: int, emp_data: schemas.EmployeeCreate):
    db_emp = get_employee(db, emp_id)
    if not db_emp:
        return None
    for key, value in emp_data.dict().items():
        setattr(db_emp, key, value)
    db.commit()
    db.refresh(db_emp)
    return db_emp

def delete_employee(db: Session, emp_id: int):
    db_emp = get_employee(db, emp_id)
    if db_emp:
        db.delete(db_emp)
        db.commit()
    return db_emp
