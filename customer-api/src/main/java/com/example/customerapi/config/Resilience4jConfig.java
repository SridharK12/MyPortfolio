package com.example.customerapi.config;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryRegistry;
import io.github.resilience4j.ratelimiter.RateLimiter;
import io.github.resilience4j.ratelimiter.RateLimiterRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

// Configuration class to set up event listeners for Resilience4j components
// These listeners log important events for monitoring and debugging
@Configuration
public class Resilience4jConfig {

    private static final Logger log = LoggerFactory.getLogger(Resilience4jConfig.class);

    // Configure Circuit Breaker event listeners
    // Logs state transitions (CLOSED -> OPEN -> HALF_OPEN) and failure events
    @Bean
    public CircuitBreaker customerStatusCircuitBreaker(CircuitBreakerRegistry circuitBreakerRegistry) {
        
        CircuitBreaker circuitBreaker = circuitBreakerRegistry.circuitBreaker("customerStatusCircuitBreaker");

        // Event listener for state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> {
                log.warn("Circuit Breaker State Transition: From {} to {} for: {}",
                        event.getStateTransition().getFromState(),
                        event.getStateTransition().getToState(),
                        circuitBreaker.getName());
            });

        // Event listener when circuit breaker opens (too many failures detected)
        circuitBreaker.getEventPublisher()
            .onError(event -> {
                log.error("Circuit Breaker Error Event: {} - Duration: {}ms",
                        event.getThrowable().getMessage(),
                        event.getElapsedDuration().toMillis());
            });

        // Event listener when circuit breaker records a successful call
        circuitBreaker.getEventPublisher()
            .onSuccess(event -> {
                log.debug("Circuit Breaker Success Event - Duration: {}ms",
                        event.getElapsedDuration().toMillis());
            });

        // Event listener when call is rejected due to open circuit
        circuitBreaker.getEventPublisher()
            .onCallNotPermitted(event -> {
                log.warn("Circuit Breaker OPEN - Call rejected for: {}", circuitBreaker.getName());
            });

        return circuitBreaker;
    }

    // Configure Retry event listeners
    // Logs retry attempts and final success/failure
    @Bean
    public Retry customerStatusRetry(RetryRegistry retryRegistry) {
        
        Retry retry = retryRegistry.retry("customerStatusRetry");

        // Event listener when retry attempt occurs
        retry.getEventPublisher()
            .onRetry(event -> {
                log.info("Retry Attempt #{} for: {} - Reason: {}",
                        event.getNumberOfRetryAttempts(),
                        retry.getName(),
                        event.getLastThrowable().getMessage());
            });

        // Event listener when all retry attempts are exhausted
        retry.getEventPublisher()
            .onError(event -> {
                log.error("All Retry Attempts Failed for: {} - Total Attempts: {}",
                        retry.getName(),
                        event.getNumberOfRetryAttempts());
            });

        // Event listener when retry succeeds (after one or more retries)
        retry.getEventPublisher()
            .onSuccess(event -> {
                log.info("Retry Succeeded for: {} - After {} attempts",
                        retry.getName(),
                        event.getNumberOfRetryAttempts());
            });

        return retry;
    }

    // Configure Rate Limiter event listeners
    // Logs when requests are rate limited
    @Bean
    public RateLimiter customerStatusRateLimiter(RateLimiterRegistry rateLimiterRegistry) {
        
        RateLimiter rateLimiter = rateLimiterRegistry.rateLimiter("customerStatusRateLimiter");

        // Event listener when request acquires permission from rate limiter
        rateLimiter.getEventPublisher()
            .onSuccess(event -> {
                log.debug("Rate Limiter Permission Acquired for: {}", rateLimiter.getName());
            });

        // Event listener when request is rejected due to rate limit exceeded
        rateLimiter.getEventPublisher()
            .onFailure(event -> {
                log.warn("Rate Limiter REJECTED Request for: {} - Rate limit exceeded",
                        rateLimiter.getName());
            });

        return rateLimiter;
    }
}
