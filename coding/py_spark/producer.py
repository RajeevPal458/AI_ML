import json
import time
from kafka import KafkaProducer
import requests
from employee import Employee
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

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

try:
    while True:
        employees: List[Employee] = get_all_employees()
        ##emp_dict = [Employee(**emp) for emp in employees]
        for emp in employees:
            #emp_dict = Employee(**emp)
            producer.send("employee-topic", asdict(emp))
            print(f"Produced: {emp}")
            time.sleep(2)  # simulate delay
finally:
    producer.flush()
