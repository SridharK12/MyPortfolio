# Testing Guide - Resilience4j Features

## Prerequisites
- Customer API running on port 8088
- Customer Status Service running (for positive tests) or stopped (for failure tests)
- curl or Postman for testing

---

## Test 1: Normal Operation (Happy Path)

**Ensure customer-status-service is running**

### Test Request
```bash
curl -X GET http://localhost:8088/api/customers/1/with-status
```

### Expected Response
```json
{
  "customerId": 1,
  "customerName": "John Doe",
  "customerDob": "1990-01-15",
  "status": "ACTIVE"
}
```

### Expected Logs
```
INFO - Before calling CustomerStatus API for Customer Id: 1
DEBUG - Circuit Breaker Success Event - Duration: 150ms
INFO - Received status from CustomerStatus API: ACTIVE
```

---

## Test 2: Circuit Breaker - Simulating Service Failure

**Stop customer-status-service before running these tests**

### Step 1: Make 5 consecutive requests to open the circuit
```bash
# Request 1
curl -X GET http://localhost:8088/api/customers/1/with-status

# Request 2
curl -X GET http://localhost:8088/api/customers/1/with-status

# Request 3
curl -X GET http://localhost:8088/api/customers/1/with-status

# Request 4
curl -X GET http://localhost:8088/api/customers/1/with-status

# Request 5
curl -X GET http://localhost:8088/api/customers/1/with-status
```

### Expected Behavior
- Each request will attempt 3 times (retry mechanism)
- After 5 failed requests, circuit breaker will OPEN
- Subsequent requests return immediately from fallback

### Expected Response (after circuit opens)
```json
{
  "customerId": 1,
  "customerName": "John Doe",
  "customerDob": "1990-01-15",
  "status": "UNAVAILABLE"
}
```

### Expected Logs
```
INFO - Retry Attempt #1 for: customerStatusRetry
INFO - Retry Attempt #2 for: customerStatusRetry
INFO - Retry Attempt #3 for: customerStatusRetry
ERROR - All Retry Attempts Failed for: customerStatusRetry - Total Attempts: 3
WARN - Circuit Breaker State Transition: From CLOSED to OPEN
WARN - Circuit Breaker OPEN - Call rejected for: customerStatusCircuitBreaker
ERROR - Fallback method called for Customer Id: 1
```

### Step 2: Check Circuit Breaker State
```bash
curl -X GET http://localhost:8088/actuator/circuitbreakers
```

### Expected Response
```json
{
  "circuitBreakers": {
    "customerStatusCircuitBreaker": {
      "state": "OPEN",
      "failureRate": "100.0%",
      "slowCallRate": "0.0%",
      "bufferedCalls": 5,
      "failedCalls": 5
    }
  }
}
```

---

## Test 3: Circuit Breaker Recovery (OPEN → HALF_OPEN → CLOSED)

### Step 1: Wait 10 seconds for circuit to transition to HALF_OPEN
```bash
# Wait 10 seconds (configured wait-duration-in-open-state)
sleep 10
```

### Step 2: Restart customer-status-service

### Step 3: Make a test request
```bash
curl -X GET http://localhost:8088/api/customers/1/with-status
```

### Expected Behavior
- Circuit transitions to HALF_OPEN
- Allows 3 test calls (configured permitted-number-of-calls-in-half-open-state)
- If calls succeed, circuit transitions to CLOSED
- Service is back to normal operation

### Expected Logs
```
WARN - Circuit Breaker State Transition: From OPEN to HALF_OPEN
INFO - Before calling CustomerStatus API for Customer Id: 1
INFO - Received status from CustomerStatus API: ACTIVE
WARN - Circuit Breaker State Transition: From HALF_OPEN to CLOSED
```

---

## Test 4: Retry Mechanism with Exponential Backoff

**Scenario: Service intermittently fails, then recovers**

### Manual Test
Start and stop customer-status-service rapidly during request

### Observe Retry Logs
```
INFO - Retry Attempt #1 for: customerStatusRetry - Reason: Connection refused
[Wait 500ms]
INFO - Retry Attempt #2 for: customerStatusRetry - Reason: Connection refused
[Wait 1000ms]
INFO - Retry Attempt #3 for: customerStatusRetry - Reason: Connection refused
INFO - Retry Succeeded for: customerStatusRetry - After 3 attempts
```

### Timing Observation
- Attempt 1 → Fail → Wait 500ms
- Attempt 2 → Fail → Wait 1000ms (2x multiplier)
- Attempt 3 → Success
- Total time: ~1.5 seconds

---

## Test 5: Rate Limiter - Exceeding Request Limit

**Configured limit: 10 requests per second**

### Test Script (Bash)
```bash
#!/bin/bash
echo "Sending 15 requests rapidly..."
for i in {1..15}
do
  echo "Request $i"
  curl -X GET http://localhost:8088/api/customers/1/with-status &
done
wait
echo "All requests completed"
```

### Expected Behavior
- First 10 requests: Succeed immediately
- Next 5 requests: 
  - Wait up to 500ms for rate limiter permission
  - If no permission, get rejected

### Expected Logs
```
DEBUG - Rate Limiter Permission Acquired for: customerStatusRateLimiter (x10)
WARN - Rate Limiter REJECTED Request for: customerStatusRateLimiter - Rate limit exceeded (x5)
```

### Rate Limiter Metrics
```bash
curl -X GET http://localhost:8088/actuator/ratelimiters
```

### Expected Response
```json
{
  "rateLimiters": {
    "customerStatusRateLimiter": {
      "availablePermissions": 0,
      "numberOfWaitingThreads": 0
    }
  }
}
```

---

## Test 6: Actuator Health Endpoint

### Check Overall Health
```bash
curl -X GET http://localhost:8088/actuator/health
```

### Expected Response (All Services UP)
```json
{
  "status": "UP",
  "components": {
    "db": {
      "status": "UP",
      "details": {
        "database": "MySQL",
        "validationQuery": "isValid()"
      }
    },
    "customerStatusServiceHealthIndicator": {
      "status": "UP",
      "details": {
        "customer-status-service": "Available",
        "message": "Customer Status Service is healthy"
      }
    },
    "circuitBreakers": {
      "status": "UP",
      "details": {
        "customerStatusCircuitBreaker": "CLOSED"
      }
    },
    "rateLimiters": {
      "status": "UP"
    }
  }
}
```

### Expected Response (customer-status-service DOWN)
```json
{
  "status": "DOWN",
  "components": {
    "db": {
      "status": "UP"
    },
    "customerStatusServiceHealthIndicator": {
      "status": "DOWN",
      "details": {
        "customer-status-service": "Unavailable",
        "error": "Connection refused",
        "message": "Customer Status Service is not responding"
      }
    },
    "circuitBreakers": {
      "status": "UP",
      "details": {
        "customerStatusCircuitBreaker": "OPEN"
      }
    }
  }
}
```

---

## Test 7: Actuator Metrics Endpoint

### Get All Available Metrics
```bash
curl -X GET http://localhost:8088/actuator/metrics
```

### Get Circuit Breaker Specific Metrics
```bash
# Failure rate
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.circuitbreaker.failure.rate?tag=name:customerStatusCircuitBreaker"

# State (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.circuitbreaker.state?tag=name:customerStatusCircuitBreaker"

# Buffered calls
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.circuitbreaker.buffered.calls?tag=name:customerStatusCircuitBreaker"
```

### Get Retry Metrics
```bash
# Successful retries
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.retry.calls?tag=name:customerStatusRetry&tag=kind:successful_with_retry"

# Failed retries
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.retry.calls?tag=name:customerStatusRetry&tag=kind:failed_with_retry"
```

### Get Rate Limiter Metrics
```bash
# Available permissions
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.ratelimiter.available.permissions?tag=name:customerStatusRateLimiter"

# Waiting threads
curl -X GET "http://localhost:8088/actuator/metrics/resilience4j.ratelimiter.waiting.threads?tag=name:customerStatusRateLimiter"
```

---

## Test 8: Combined Scenario - Full Resilience Stack

### Scenario
1. Service is slow (triggers slow call threshold)
2. Then fails completely (triggers failure threshold)
3. Circuit opens
4. Service recovers
5. Circuit closes

### Test Steps

**Step 1: Make service slow**
Modify customer-status-service to add delay:
```java
Thread.sleep(3000); // 3 seconds (exceeds 2s slow call threshold)
```

**Step 2: Make 5 slow requests**
```bash
for i in {1..5}; do
  echo "Slow request $i"
  curl -X GET http://localhost:8088/api/customers/1/with-status
done
```

**Expected**: Circuit breaker records slow calls

**Step 3: Stop customer-status-service**

**Step 4: Make 5 more requests**
```bash
for i in {1..5}; do
  echo "Failed request $i"
  curl -X GET http://localhost:8088/api/customers/1/with-status
done
```

**Expected**: Circuit opens (50% failure + 50% slow = 100% unhealthy)

**Step 5: Verify circuit is OPEN**
```bash
curl -X GET http://localhost:8088/actuator/circuitbreakers
```

**Step 6: Wait 10 seconds and restart service**
```bash
sleep 10
# Restart customer-status-service without delay
```

**Step 7: Make recovery requests**
```bash
for i in {1..3}; do
  echo "Recovery request $i"
  curl -X GET http://localhost:8088/api/customers/1/with-status
done
```

**Expected**: Circuit transitions OPEN → HALF_OPEN → CLOSED

---

## Test 9: Load Testing with Rate Limiter

### Using Apache Bench
```bash
# Send 100 requests with 20 concurrent connections
ab -n 100 -c 20 http://localhost:8088/api/customers/1/with-status
```

### Expected Results
- Some requests succeed (rate limited to 10/sec)
- Some requests rejected or delayed
- Circuit breaker might open if too many fail

### Check Rate Limiter Status
```bash
curl -X GET http://localhost:8088/actuator/ratelimiters
```

---

## Test 10: Error Scenarios

### Test 1: Customer Not Found
```bash
curl -X GET http://localhost:8088/api/customers/999/with-status
```

**Expected Response**: 404 NOT FOUND
**Expected**: Resilience4j NOT triggered (business logic error)

### Test 2: Database Down
```bash
# Stop MySQL
curl -X GET http://localhost:8088/api/customers/1/with-status
```

**Expected Response**: 500 Internal Server Error
**Expected**: No retry (database exception not in retry list)

---

## Monitoring Commands

### Quick Status Check Script
```bash
#!/bin/bash
echo "=== Circuit Breaker Status ==="
curl -s http://localhost:8088/actuator/circuitbreakers | jq

echo "\n=== Rate Limiter Status ==="
curl -s http://localhost:8088/actuator/ratelimiters | jq

echo "\n=== Health Status ==="
curl -s http://localhost:8088/actuator/health | jq

echo "\n=== Application Info ==="
curl -s http://localhost:8088/actuator/info | jq
```

---

## Expected Performance

**Normal Operation**:
- Response time: 100-200ms
- Success rate: 100%
- No retries

**With Slow Service** (1-2s response):
- Response time: 1-2 seconds
- Success rate: 100%
- Circuit breaker records slow calls

**With Failing Service**:
- Response time: 1.5-2 seconds (3 retry attempts)
- Success rate: 0% (returns fallback)
- Circuit opens after 5 failures

**With Circuit OPEN**:
- Response time: <10ms (immediate fallback)
- Success rate: 100% (fallback response)
- No external calls made

---

## Troubleshooting

### Circuit Won't Open
- Check if minimum-number-of-calls (5) is reached
- Verify failure-rate-threshold is exceeded
- Check logs for circuit breaker events

### Retry Not Working
- Verify exception type is in retry-exceptions list
- Check max-attempts configuration
- Look for "Retry Attempt" log messages

### Rate Limiter Not Rejecting
- Verify requests exceed limit-for-period (10/sec)
- Check timeout-duration configuration
- Monitor rate limiter metrics

### Fallback Not Called
- Ensure fallback method signature matches original method
- Check fallback method name in @CircuitBreaker annotation
- Verify exception is not in ignore-exceptions list
