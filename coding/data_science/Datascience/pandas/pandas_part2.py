## practicing data fetching from url and from database 
import requests
from models.employee import Employee
from typing import List
import json
from dataclasses import asdict
import pandas as pd


url = 'http://127.0.0.1:8000/api/v1/employees/'

def get_all_employees() -> List[Employee]:
    empList: List[Employee] = []
    try:
        response = requests.get(url=url)
        #print('api call success response: ',response)
        if response.status_code==200:
            data = response.json()
            #print(json.dumps(data, indent=4))
            employees:List[Employee] = [Employee(**emp) for emp in data]
            empList = employees
            df = pd.DataFrame(data)
            print("pandas data frame are below ")
            print(df.head(2))
        else:
            print('api call failed with status_code: ',response.status_code)
        
    except Exception as e:
        print('exception raised :',e)
    return empList

def get_employee_by_id(emp_id: int) -> Employee:
    emp_response:Employee=None
    try:
        response = requests.get(url=url+str(emp_id))
        print('api call success response: ',response)
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data,indent=4))
            employee = Employee(**data)
            emp_response = employee
           
    except Exception as ex:
        print('exception raised in get_employee_by_id method ',ex)
    return emp_response

def create_employee(emp: Employee) -> Employee:
    result : Employee | None = None
    try:
        response = requests.post(url=url,json=asdict(emp))
        print("API call success response:",response)
        if response.status_code == 200:
            json_obj = response.json()
            print(json.dumps(json_obj,indent=4))
            emp_obj = Employee(**json_obj)
            result = emp_obj
        else:
            print("api call failed with response code:",response.status_code)
    except Exception as ex:
        print("exception raised when doing api call:",ex)
    
    return result
            
        

list_of_employees: List[Employee] = get_all_employees()
print("api's response ",list_of_employees)

emp_id:int = 101
employee_by_id: Employee = get_employee_by_id(emp_id)
print(f"api call with id :{emp_id}, response: {employee_by_id}")


emp:Employee = Employee('Neenaaa','pal','neenaaa@jina.com','7777788880','1989-09-20',19000.10,80,"https://www.oracle.com/webfolder/technetwork/jet/Images/dvt/2.png",1)
api_response:Employee = create_employee(emp)
print("created employee :",api_response)

###############################################################
###############################################################

