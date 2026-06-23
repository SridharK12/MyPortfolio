# idapractice — Payment Processing Platform

A production-grade payment microservices platform built with Spring Boot, Kafka, MySQL and the ELK stack. Developed as a portfolio project to demonstrate enterprise Java architecture patterns used in BFSI (Banking, Financial Services and Insurance) systems.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client / React UI                        │
│                        (port 3000)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ REST
┌─────────────────────────▼───────────────────────────────────────┐
│                   idapractice (port 8080)                        │
│                   Payment Service                                │
│                                                                  │
│  PaymentController → PaymentService → PaymentOutboxPublisher    │
│                              │                                   │
│                      payment_outbox table                        │
│                         (MySQL - idadb)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Kafka (payment-approved)
┌─────────────────────────▼───────────────────────────────────────┐
│               risk-scoring-service (port 8081)                   │
│                                                                  │
│         PaymentApprovedConsumer → RiskAssessmentRepository      │
│                              │                                   │
│                     risk_assessments table                       │
│                        (MySQL - riskdb)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Port | Description |
|---|---|---|
| idapractice | 8080 | Payment initiation, authorization, outbox publisher |
| risk-scoring-service | 8081 | Consumes approved payments, persists for risk review |
| MySQL | 3306 | Two schemas — idadb and riskdb |
| Kafka | 29092 | Event streaming between services |
| Zookeeper | 2181 | Kafka coordination |
| Elasticsearch | 9200 | Log storage |
| Logstash | 5044 | Log pipeline |
| Kibana | 5601 | Log visualization |

---

## Key Design Patterns

### Transactional Outbox Pattern
Payment state changes write to a `payment_outbox` table **within the same database transaction** as the payment record. A separate `PaymentOutboxPublisher` polls the outbox and publishes to Kafka. This eliminates the dual-write problem — if Kafka is down, no events are lost.

### Idempotency
Every payment creation accepts an optional `X-Idempotency-Key` header. Duplicate requests with the same key return the original payment without creating a new record. Safe for client retries.

### Optimistic Locking
The `Payment` entity carries a `@Version` field. Concurrent updates to the same payment result in an `HTTP 409` rather than a silent lost update.

### Microservice Data Isolation
Each service owns its own database schema (`idadb` and `riskdb`). Services share data exclusively through Kafka events — no shared tables, no cross-service joins.

### Dead Letter Topic (DLT)
Kafka consumer failures retry 3 times with a 1-second backoff. After exhausting retries the message is routed to `<topic>.DLT` for manual inspection and replay.

---

## Payment Lifecycle

```
POST /v1/payments
        │
        ▼
   PENDING ──── PUT /v1/payments/{id} ──── (update amount, accounts, remarks)
        │
        ▼
POST /v1/payments/{id}/authorization
        │
   ┌────┴────┐
   ▼         ▼
APPROVED   REJECTED
   │
   ▼
Kafka: payment-approved
   │
   ▼
risk-scoring-service persists RiskAssessment
(status: PENDING_REVIEW → future ML scoring)

DELETE /v1/payments/{id}
        │
        ▼
   CANCELLED
```

---

## API Reference

### Payment Service (port 8080)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/payments` | Create a PENDING payment |
| GET | `/v1/payments` | List payments (filter by status, fromAccount) |
| GET | `/v1/payments/{id}` | Get payment by ID |
| PUT | `/v1/payments/{id}` | Update a PENDING payment |
| DELETE | `/v1/payments/{id}` | Cancel a PENDING payment |
| POST | `/v1/payments/{id}/authorization` | Approve or reject a payment |
| GET | `/actuator/health` | Health probe |

#### Create Payment
```bash
curl -X POST http://localhost:8080/v1/payments \
  -H "Content-Type: application/json" \
  -H "X-Idempotency-Key: uuid-here" \
  -d '{
    "fromAccount": "ACC001",
    "toAccount": "ACC002",
    "amount": 15000.00,
    "remarks": "Invoice payment"
  }'
```

#### Authorize Payment
```bash
curl -X POST http://localhost:8080/v1/payments/1/authorization \
  -H "Content-Type: application/json" \
  -d '{
    "status": "APPROVED",
    "remarks": "Verified OK"
  }'
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Java 17 |
| Framework | Spring Boot 4.1.0 |
| ORM | Spring Data JPA + Hibernate |
| Database | MySQL 8.0 |
| Migrations | Flyway |
| Messaging | Apache Kafka |
| Observability | ELK Stack (Elasticsearch, Logstash, Kibana) |
| Frontend | React + Vite + Recharts |
| Containerization | Docker + Docker Compose |

---

## Running Locally

### Prerequisites
- Docker Desktop
- Java 17
- Maven 3.9+

### Start the full stack

```bash
# Clone the repo
git clone https://github.com/SridharK12/MyPortfolio.git
cd MyPortfolio/idapractice

# Start all infrastructure + services
docker-compose up --build
```

### Access points
- Payment API: http://localhost:8080/v1/payments
- Risk Service: http://localhost:8081/actuator/health
- Kibana: http://localhost:5601
- Kafka: localhost:29092

### Running the Spring Boot apps locally (without Docker)

```bash
# Terminal 1 — start infrastructure only
docker-compose up zookeeper kafka mysql elasticsearch logstash kibana

# Terminal 2 — start idapractice
cd idapractice
mvn spring-boot:run

# Terminal 3 — start risk-scoring-service
cd risk-scoring-service
mvn spring-boot:run
```

---

## Project Structure

```
idapractice/
├── src/main/java/com/idapractice/idapractice/
│   ├── controller/        # REST endpoints
│   ├── service/           # Business logic
│   ├── entity/            # JPA entities
│   ├── dto/               # Request / response objects
│   ├── outbox/            # Transactional outbox publisher
│   ├── repository/        # Spring Data repositories
│   ├── config/            # Kafka, Jackson, Scheduling config
│   ├── enums/             # PaymentStatus, AccountType
│   └── exception/         # GlobalExceptionHandler, custom exceptions
├── src/main/resources/
│   ├── db/migration/      # Flyway SQL migrations
│   └── application.properties
├── UI/payment-ui/         # React frontend
├── docker-compose.yml     # Full stack deployment
└── Dockerfile

risk-scoring-service/
├── src/main/java/com/idapractice/riskscoring/
│   ├── consumer/          # Kafka listener (payment-approved)
│   ├── entity/            # RiskAssessment
│   ├── dto/               # PaymentApprovedEvent
│   ├── repository/        # RiskAssessmentRepository
│   ├── config/            # KafkaConsumerConfig with DLT
│   └── enums/             # RiskStatus
├── src/main/resources/
│   ├── db/migration/      # V1__create_risk_assessments_table.sql
│   └── application.properties
└── Dockerfile
```

---

## Roadmap

- [ ] ML-based fraud scoring in risk-scoring-service
- [ ] Saga pattern for payment execution (debit/credit with compensation)
- [ ] JWT authentication
- [ ] API rate limiting
- [ ] Kubernetes deployment manifests

---

## Author

Sridhar K — Senior Engineering Manager with deep BFSI and payments domain experience.
