from db.load_employees_data import EmployeesDataUploader
from models.employee import Employee,EmployeeResponse,EmployeeDto
from sqlalchemy.orm import Session

class EmployeeRepository:
    # declare a varriable to hold list of employees and also add the Employee type 
    
    def get_all_employees(self, db: Session):
        return db.query(Employee).all()

    def get_employee(self, db: Session, employee_id: int):
        result = Employee()
        try:
            result= db.query(Employee).filter(Employee.id == employee_id).first()
            print(f"Employee with ID {employee_id}: {result}")
        except Exception as e:
            print(f"Error fetching employee with ID {employee_id}: {e}")
        return result

    def create_employee(self, db: Session, employee: Employee):
        db.add(employee)
        db.commit()
        db.refresh(employee)
        return employee

    def save_all(self, db: Session, employees: list[Employee]):
        try:
            db.add_all(employees)
            db.commit()
            for emp in employees:
                db.refresh(emp)
        except Exception as e:
            raise e
        return employees

    def update_employee(self, db: Session, employee_id: int, updated_data: Employee):
        try:
            employee = db.query(Employee).filter(Employee.id == employee_id).first()
        except Exception as e:
            print(f'Exception occurred while querying employee: {e}')
            employee = None

        if not employee:
            updated_data.id = employee_id  # ✅ important: assign the ID if inserting new
            merged_employee = db.merge(updated_data)  # ✅ merge returns the managed object
            db.commit()
            db.refresh(merged_employee)  # ✅ refresh to load DB-generated fields
            print('merge successful!')
            return merged_employee

        # ✅ update existing fields
        employee.DEPARTMENT_ID = updated_data.DEPARTMENT_ID
        employee.EMAIL = updated_data.EMAIL
        employee.FIRST_NAME = updated_data.FIRST_NAME
        employee.LAST_NAME = updated_data.LAST_NAME
        employee.Image = updated_data.Image
        employee.PHONE_NUMBER = updated_data.PHONE_NUMBER
        employee.HIRE_DATE = updated_data.HIRE_DATE
        employee.SALARY = updated_data.SALARY

        db.commit()
        db.refresh(employee)
        return employee

    def delete_employee(self, db: Session, employee_id: int):
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if employee:
            db.delete(employee)
            db.commit()
        return employee

