from models.employee import Employee
import json
class EmployeesDataUploader:
    # Class-level variable to hold the single instance
    _instance = None


    # __new__ is responsible for creating and returning the actual object
    def __new__(cls):
        # Check if an instance already exists
        if cls._instance is None:
            print("Creating a new EmployeesDataUploader instance")
            # If not, create one using the superclass's __new__ method
            cls._instance = super(EmployeesDataUploader, cls).__new__(cls)
        else:
            print("Returning existing EmployeesDataUploader instance")
        
        # Return the same instance every time
        return cls._instance

    # Optional: your normal initializer method
    def __init__(self):
        print("Initializing EmployeesDataUploader instance (called every time, but only one object exists)")


    def upload_json_data(self):
        # upload json file employees.json in current directory
        print("Uploading JSON data... (this is just a placeholder for actual upload logic)")
        with open('employees.json', 'r') as file:
            data = json.load(file)
        print("Data uploaded successfully.")
        # Map each dictionary to an Employee object
        # ✅ Extract the list under the "employees" key
        employee_dicts = data.get("employees", [])
        self.__employeeList = [Employee(**item) for item in employee_dicts]


    def get_employees(self) -> list[Employee]:
        return self.__employeeList
    
    def set_employees(self, employees: list[Employee]):
        self.__employeeList = employees


    


# 🔁 Test the EmployeesDataUploader

obj1 = EmployeesDataUploader()
obj2 = EmployeesDataUploader()

# Check if both objects are actually the same (EmployeesDataUploader)
print(obj1 is obj2)  # Output: True

