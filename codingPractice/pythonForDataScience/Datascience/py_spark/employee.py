from typing import Optional
from datetime import date
from dataclasses import dataclass

@dataclass
class Employee:
    FIRST_NAME: str
    LAST_NAME: str
    EMAIL: str
    PHONE_NUMBER: str
    HIRE_DATE: date
    SALARY: float
    DEPARTMENT_ID: int
    Image: Optional[str]
    id: int
    