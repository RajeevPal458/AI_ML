# controllers/employee_controller.py

from fastapi import APIRouter, Depends, HTTPException
from models.employee import Employee, EmployeeDto, EmployeeResponse
from services.employee_service import EmployeeService
from repositories.employee_repository import EmployeeRepository
from sqlalchemy.orm import Session
from dependencies.di import get_db
employee_router = APIRouter()

# --- Dependency resolver ---
service =EmployeeService(employee_repository=EmployeeRepository())


@employee_router.get(
    "/employees",
    response_model=list[EmployeeResponse],
    summary="Get all employees",
    description="Returns a list of all employees from the database."
)
async def get_all_employees(db: Session = Depends(get_db)):
    return await service.get_all_employees(db)


@employee_router.get(
    "/employees/{employee_id}",
    response_model=EmployeeResponse,
    summary="Get an employee by ID",
    description="Returns the employee details for a given employee ID."
)
async def get_employee_by_id(
    employee_id: int,db: Session = Depends(get_db)
):
    return await service.get_employee_by_id(db,employee_id)


@employee_router.post(
    "/employees",
    response_model=EmployeeResponse,
    summary="Create a new employee",
    description="Creates a new employee with the provided data."
)
async def create_employee(
    employee_data: EmployeeDto, db: Session = Depends(get_db)
):
    return await service.create_employee(db,employee_data)


@employee_router.put(
    "/employees/{employee_id}",
    response_model=EmployeeResponse,
    summary="Update employee data",
    description="Updates an existing employee's data based on the employee ID."
)
async def update_employee(
    employee_id: int,
    employee_data: EmployeeDto,
    db: Session = Depends(get_db),
):
    return await service.update_employee(db,employee_id, employee_data)


@employee_router.delete(
    "/employees/{employee_id}",
    response_model=EmployeeResponse,
    summary="Delete an employee",
    description="Deletes the employee with the specified ID."
)
async def delete_employee(
    employee_id: int, db: Session = Depends(get_db),
):
    return await service.delete_employee(db,employee_id)
