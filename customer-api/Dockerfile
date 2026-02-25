# Base image with Java 17 runtime
FROM eclipse-temurin:17-jre

# Working directory inside container
WORKDIR /app

# Copy Spring Boot JAR
COPY target/customer-api-1.0.0.jar app.jar

# Expose application port
EXPOSE 8080

# Run the application
ENTRYPOINT ["java","-jar","app.jar"]
