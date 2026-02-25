# Customer Status API - Runnable Spring Boot Project (MySQL)

## Database
This project is configured to use **MySQL** (database name: management).

Connection is set in `application.properties`:
- URL: jdbc:mysql://localhost:3306/management
- Username: sridhar
- Password: srid1234
- Server port: 8090

## How to run
1. Ensure JDK 17 and Maven are installed.
2. Create the database and user in MySQL (example):
   ```
   CREATE DATABASE management CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   CREATE USER 'sridhar'@'localhost' IDENTIFIED BY 'srid1234';
   GRANT ALL PRIVILEGES ON management.* TO 'sridhar'@'localhost';
   FLUSH PRIVILEGES;
   ```
3. Optionally run the schema.sql manually or let Spring run it.
4. From project root:
   ```
   mvn spring-boot:run
   ```
5. Test endpoint:
   `GET http://localhost:8090/api/customers/1001/status`

H2 is NOT used. The `schema.sql` and `data.sql` are MySQL-compatible.
