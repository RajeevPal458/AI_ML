# main.py

from fastapi import FastAPI
from controllers.employee_controller import employee_router

app = FastAPI()

# Register router with FastAPI
app.include_router(employee_router, prefix="/api/v1", tags=["employees"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Employee Management API!"}

@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "Service is running smoothly!"}

@app.get("/version")
async def version():
    return {"version": "1.0.0", "description": "Employee Management API Version 1.0.0"}
