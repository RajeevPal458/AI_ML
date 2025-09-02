# models/employee_dto.py

from pydantic import BaseModel
from typing import Optional
from datetime import date
from db.database import Base
from sqlalchemy import Column,Integer,String,Date,Float,URL

class Employee(Base):
    
    __tablename__ =  "employees"
    id = Column(Integer,primary_key=True,index=True)
    FIRST_NAME = Column(String,nullable=False)
    LAST_NAME = Column(String)
    EMAIL = Column(String,nullable=False,unique=True)
    PHONE_NUMBER= Column(String,nullable=False,unique=True)
    HIRE_DATE= Column(Date,nullable=False)
    SALARY = Column(Float)
    DEPARTMENT_ID= Column(Integer)
    Image = Column(String)


# Pydantic models for validation & response
class EmployeeDto(BaseModel):
    FIRST_NAME: str
    LAST_NAME: str
    EMAIL: str
    PHONE_NUMBER: str
    HIRE_DATE: date
    SALARY: float
    DEPARTMENT_ID: int
    Image: Optional[str]

class EmployeeResponse(EmployeeDto):
    id: int

    model_config = {
        "from_attributes": True  # ✅ correct for v2
    }  # enables SQLAlchemy -> Pydantic conversion