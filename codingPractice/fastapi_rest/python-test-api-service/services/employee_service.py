
from repositories.employee_repository import EmployeeRepository
from db.load_employees_data import EmployeesDataUploader
from sqlalchemy.orm import Session
from models.employee import Employee, EmployeeDto, EmployeeResponse
class EmployeeService:
    
    def __init__(self, employee_repository: EmployeeRepository):
        self.employee_repository = employee_repository
        self.employeesDataUploader = EmployeesDataUploader()
        self.employeesDataUploader.upload_json_data()
        self.flag:True = bool
        
    async def get_all_employees(self,db:Session):
        # try:
        #     if(self.flag):
        #         self.employee_repository.save_all(db,self.employeesDataUploader.get_employees())
        #         self.flag = False
        # except Exception as e:
        #     print('exception occures while saving')
        return self.employee_repository.get_all_employees(db)

    async def get_employee_by_id(self,db:Session, employee_id):
        return self.employee_repository.get_employee(db,employee_id)

    async def create_employee(self,db:Session, employee_data):
        employee = Employee(**employee_data.dict())
        return self.employee_repository.create_employee(db,employee)

    async def update_employee(self,db:Session, employee_id, employee_data):
        employee = Employee(**employee_data.dict())
        return self.employee_repository.update_employee(db,employee_id, employee)

    async def delete_employee(self,db:Session, employee_id):
        
        return self.employee_repository.delete_employee(db,employee_id)