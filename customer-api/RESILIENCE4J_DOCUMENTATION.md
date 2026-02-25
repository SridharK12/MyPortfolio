# Customer API - Resilience4j Integration

## Overview
This application has been enhanced with Resilience4j to provide robust fault tolerance when calling the external customer-status-service. The following resilience patterns have been implemented:

1. **Circuit Breaker** - Prevents cascading failures
2. **Retry** - Automatic retry with exponential backoff
3. **Rate Limiter** - Controls request rate to external service
4. **Actuator** - Monitoring and health check endpoints

---

## Features Added

### 1. Circuit Breaker
**Purpose**: Prevents the application from repeatedly calling a failing service, giving it time to recover.

**Configuration** (application.properties):
- Failure threshold: 50% (circuit opens if half the calls fail)
- Slow call threshold: 50% (circuit opens if half the calls are slow)
- Slow call duration: 2 seconds
- Sliding window: 10 calls (count-based)
- Minimum calls: 5 (before calculating failure rate)
- Wait duration in OPEN state: 10 seconds
- Half-open permitted calls: 3 (test calls when recovering)

**Circuit States**:
- **CLOSED**: Normal operation, all calls go through
- **OPEN**: Service is failing, calls are rejected immediately, fallback is invoked
- **HALF_OPEN**: Testing if service has recovered, allows limited calls

**Fallback Method**: `getCustomerDetailsWithStatusFallback()`
- Returns customer data with status="UNAVAILABLE" when circuit is open
- Prevents complete failure, provides degraded functionality

### 2. Retry Mechanism
**Purpose**: Automatically retries failed calls due to transient network issues.

**Configuration**:
- Max attempts: 3
- Wait duration: 500ms (initial)
- Exponential backoff enabled: true
- Backoff multiplier: 2x (500ms, 1000ms, 2000ms)
- Retry exceptions: ResourceAccessException, IOException
- Ignore exceptions: BusinessException (don't retry business logic errors)

**Behavior**:
- 1st attempt fails → wait 500ms → retry
- 2nd attempt fails → wait 1000ms → retry
- 3rd attempt fails → give up, invoke fallback

### 3. Rate Limiter
**Purpose**: Protects the external service from being overwhelmed with too many requests.

**Configuration**:
- Limit period: 1 second
- Requests per period: 10
- Timeout: 500ms (wait time for permission)

**Behavior**:
- Allows maximum 10 requests per second to customer-status-service
- If limit exceeded, request waits up to 500ms for permission
- If still no permission, request is rejected

### 4. Spring Boot Actuator
**Purpose**: Provides production-ready monitoring and management endpoints.

**Exposed Endpoints**:
- `/actuator/health` - Application and dependency health status
- `/actuator/info` - Application information
- `/actuator/metrics` - Application metrics
- `/actuator/circuitbreakers` - Circuit breaker status and metrics
- `/actuator/ratelimiters` - Rate limiter metrics
- `/actuator/retries` - Retry statistics
- `/actuator/threaddump` - Thread information
- `/actuator/env` - Environment properties
- `/actuator/beans` - Spring beans information

**Health Checks**:
- Database connectivity
- Customer-status-service availability (custom health indicator)
- Circuit breaker states
- Rate limiter status

---

## Code Changes

### 1. pom.xml Dependencies Added
```xml
<!-- Actuator for monitoring -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>

<!-- Resilience4j for circuit breaker, retry, rate limiter -->
<dependency>
    <groupId>io.github.resilience4j</groupId>
    <artifactId>resilience4j-spring-boot3</artifactId>
</dependency>

<!-- AOP support for Resilience4j annotations -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>
```

### 2. CustomerService.java Changes
**Added Annotations**:
```java
@CircuitBreaker(name = "customerStatusCircuitBreaker", fallbackMethod = "getCustomerDetailsWithStatusFallback")
@Retry(name = "customerStatusRetry")
@RateLimiter(name = "customerStatusRateLimiter")
public CustomerDetailsWithStatusResponse getCustomerDetailsWithStatus(Long id)
```

**Added Fallback Method**:
```java
public CustomerDetailsWithStatusResponse getCustomerDetailsWithStatusFallback(Long id, Throwable throwable) {
    // Returns customer data with status="UNAVAILABLE"
    // Called when circuit is open or all retries fail
}
```

### 3. New Files Created

**CustomerStatusServiceHealthIndicator.java**
- Custom health indicator
- Checks if customer-status-service is available
- Appears in `/actuator/health` endpoint

**Resilience4jConfig.java**
- Event listeners for circuit breaker, retry, and rate limiter
- Logs state transitions and important events
- Helps in monitoring and debugging

---

## Testing the Features

### 1. Test Circuit Breaker

**Scenario**: Simulate service failure
```bash
# Stop customer-status-service
# Make multiple requests (at least 5)
curl http://localhost:8088/api/customers/1/with-status

# After 5 failures, circuit will OPEN
# Subsequent requests will return immediately with fallback response
# Status will be "UNAVAILABLE"
```

**Check Circuit State**:
```bash
curl http://localhost:8088/actuator/circuitbreakers
```

### 2. Test Retry

**Scenario**: Temporary network glitch
```bash
# Make a request while service is starting up
curl http://localhost:8088/api/customers/1/with-status

# Check logs - you'll see retry attempts:
# "Retry Attempt #1"
# "Retry Attempt #2"
# "Retry Attempt #3"
```

### 3. Test Rate Limiter

**Scenario**: Make rapid requests
```bash
# Send 15 requests rapidly (limit is 10/second)
for i in {1..15}; do
  curl http://localhost:8088/api/customers/1/with-status &
done

# First 10 will succeed
# Remaining 5 will be rate limited
# Check logs for "Rate Limiter REJECTED Request"
```

### 4. Test Actuator Endpoints

```bash
# Check overall health
curl http://localhost:8088/actuator/health

# Check circuit breaker metrics
curl http://localhost:8088/actuator/circuitbreakers

# Check application metrics
curl http://localhost:8088/actuator/metrics

# Check all available endpoints
curl http://localhost:8088/actuator
```

---

## Execution Flow

When a request comes to `/api/customers/{id}/with-status`:

```
1. Controller receives request
   ↓
2. Rate Limiter checks: Can we make this request?
   - YES → Continue
   - NO → Reject (wait or fail)
   ↓
3. Circuit Breaker checks: Is circuit CLOSED?
   - CLOSED → Continue
   - OPEN/HALF_OPEN → Call fallback immediately
   ↓
4. Retry mechanism wraps the actual call
   - Attempt 1 → Fails → Wait 500ms
   - Attempt 2 → Fails → Wait 1000ms
   - Attempt 3 → Fails → Circuit records failure
   ↓
5. Circuit Breaker updates failure count
   - If threshold reached → Circuit opens
   ↓
6. If all fails → Fallback method returns "UNAVAILABLE" status
```

---

## Monitoring in Production

### Important Metrics to Watch

1. **Circuit Breaker Metrics**:
   - State (CLOSED/OPEN/HALF_OPEN)
   - Failure rate
   - Slow call rate
   - Number of buffered calls

2. **Retry Metrics**:
   - Number of retry attempts
   - Success rate after retry
   - Failed retry count

3. **Rate Limiter Metrics**:
   - Available permissions
   - Waiting threads
   - Number of successful/failed acquisitions

### Log Messages to Monitor

- `Circuit Breaker State Transition: From CLOSED to OPEN`
- `Retry Attempt #N`
- `Rate Limiter REJECTED Request`
- `Fallback method called for Customer Id`

---

## Benefits

1. **Fault Tolerance**: Service continues to work even when customer-status-service is down
2. **Automatic Recovery**: Circuit breaker automatically tests if service has recovered
3. **Resource Protection**: Rate limiter prevents overwhelming the external service
4. **Transient Failure Handling**: Retry mechanism handles temporary network issues
5. **Observability**: Actuator endpoints provide real-time monitoring
6. **Graceful Degradation**: Fallback provides reduced functionality instead of complete failure

---

## Configuration Tuning

Adjust these values based on your requirements:

**For More Aggressive Circuit Breaking**:
- Reduce `failure-rate-threshold` (e.g., 30%)
- Reduce `minimum-number-of-calls` (e.g., 3)
- Reduce `wait-duration-in-open-state` (e.g., 5s)

**For More Retries**:
- Increase `max-attempts` (e.g., 5)
- Increase `wait-duration` (e.g., 1000ms)

**For Higher Throughput**:
- Increase `limit-for-period` (e.g., 20 requests/second)

---

## Best Practices

1. Always provide meaningful fallback responses
2. Monitor circuit breaker state transitions in production
3. Set appropriate timeout values for external service calls
4. Log all retry attempts for debugging
5. Use actuator endpoints for health checks in load balancers
6. Configure alerts based on circuit breaker state changes
7. Test resilience patterns in staging environment before production

---

## Dependencies Version
- Spring Boot: 3.2.0
- Resilience4j: Managed by Spring Cloud BOM 2023.0.4
- Java: 17
