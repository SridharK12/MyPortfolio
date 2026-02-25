# Customer API - Spring Boot CRUD Application

## Prerequisites
- MySQL database named `customer` must be created
- MySQL user: sridhar with password: srid1234

## API Endpoints

### Get All Customers
```
GET http://localhost:8080/api/customers
```

### Get Customer by ID
```
GET http://localhost:8080/api/customers/{id}
```

### Create Customer
```
POST http://localhost:8080/api/customers
Content-Type: application/json

{
  "customerName": "John Doe",
  "customerDob": "1990-05-15",
  "modifiedBy": "admin"
}
```

### Update Customer
```
PUT http://localhost:8080/api/customers/{id}
Content-Type: application/json

{
  "customerName": "John Updated",
  "customerDob": "1990-05-15",
  "modifiedBy": "admin"
}
```

### Delete Customer
```
DELETE http://localhost:8080/api/customers/{id}
```

## Run the Application
```bash
mvn spring-boot:run
```
