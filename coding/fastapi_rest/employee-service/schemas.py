from pydantic import BaseModel

class EmployeeBase(BaseModel):
    name: str
    age: int
    bonus: float
    salary: float
    department: str
    salary_2025: float
    gender: str
    salary_rank: int

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeOut(EmployeeBase):
    id: int

    class Config:
        orm_mode = True
