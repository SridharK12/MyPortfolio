package com.example.customerapi.service;

import com.example.customerapi.dto.CustomerDetailsWithStatusResponse;
import com.example.customerapi.dto.CustomerStatusResponse;
import com.example.customerapi.entity.Customer;
import com.example.customerapi.repository.CustomerRepository;
import com.example.customerapi.mapper.CustomerMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.util.StringUtils;
import com.example.customerapi.exception.BusinessException;

// Resilience4j imports for circuit breaker, retry, and rate limiter
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;


import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@Service
public class CustomerService {

    @Autowired
    private CustomerRepository customerRepository;

    @Autowired
    private RestTemplate restTemplate;   // <-- DIRECT API CALL WILL USE THIS
    
    private static final String STATUS_URL =
            "http://customer-status-service/api/customers/{id}";
    
    private static final Logger log = LoggerFactory.getLogger(CustomerService.class);


    public List<Customer> getAllCustomers() {
        return customerRepository.findAll();
    }

    public Optional<Customer> getCustomerById(Long id) {
        return customerRepository.findById(id);
    }

    public Customer createCustomer(Customer customer) {
    	
        if (!StringUtils.hasText(customer.getCustomerState())) {
            throw new BusinessException(
                    "STATE_CODE_MISSING",
                    "State code must not be null or empty",
                    HttpStatus.BAD_REQUEST
            );
        }
        customer.setModificationDate(LocalDate.now());
        return customerRepository.save(customer);
    }
    
    public Customer updateCustomer(Long id, Customer customerDetails) {
        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Customer not found"));
        
        log.info("Updating Customer Record for Customer Id {}", id);

        customer.setCustomerName(customerDetails.getCustomerName());
        customer.setCustomerDob(customerDetails.getCustomerDob());
        customer.setModifiedBy(customerDetails.getModifiedBy());
        customer.setModificationDate(LocalDate.now());

        return customerRepository.save(customer);
    }

    public boolean deleteCustomer(Long id) {
        if (!customerRepository.existsById(id)) {
            return false;
        }
        customerRepository.deleteById(id);
        return true;
    }

    // ---------------------------------------------------------
    // NEW METHOD: RETURN CUSTOMER + STATUS WITH RESILIENCE4J
    // ---------------------------------------------------------
    // @CircuitBreaker: Prevents cascading failures by opening circuit when threshold is reached
    //                  If circuit is OPEN, fallback method is called immediately without calling external service
    // @Retry: Automatically retries failed calls based on configuration (3 attempts with exponential backoff)
    // @RateLimiter: Controls the rate of calls to prevent overwhelming the external service (10 req/sec)
    // Execution order: RateLimiter -> CircuitBreaker -> Retry -> Actual Method Call
    @CircuitBreaker(name = "customerStatusCircuitBreaker", fallbackMethod = "getCustomerDetailsWithStatusFallback")
    @Retry(name = "customerStatusRetry")
    @RateLimiter(name = "customerStatusRateLimiter")
    public CustomerDetailsWithStatusResponse getCustomerDetailsWithStatus(Long id) {

        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Customer not found"));
        
        log.trace("In customer status Before calling CustomerStatus API for Customer Id: {}", id);

        // --- CALL CUSTOMER STATUS SERVICE ---
        // This call is protected by circuit breaker, retry, and rate limiter
        CustomerStatusResponse statusResponse = restTemplate.getForObject(
                STATUS_URL,
                CustomerStatusResponse.class,
                id
        );

        String status = (statusResponse != null) ? statusResponse.getStatus() : "UNKNOWN";
        
        log.info("Received status from CustomerStatus API: {}", status);

//        String status="Active";
        // --- MAP TO DTO ---
        return CustomerMapper.mapToDto(customer, status);
    }
    
    // ---------------------------------------------------------
    // FALLBACK METHOD FOR CIRCUIT BREAKER
    // ---------------------------------------------------------
    // This method is called when:
    // 1. Circuit breaker is in OPEN state (too many failures detected)
    // 2. All retry attempts are exhausted
    // 3. Rate limiter rejects the request
    // 4. Any exception occurs that is not ignored
    // The method signature must match the original method with an additional Throwable parameter
    public CustomerDetailsWithStatusResponse getCustomerDetailsWithStatusFallback(Long id, Throwable throwable) {
        
        log.error("Fallback method called for Customer Id: {}. Reason: {}", id, throwable.getMessage());
        
        // Retrieve customer details from database
        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Customer not found"));
        
        // Return customer details with a default/fallback status since external service is unavailable
        String fallbackStatus = "UNAVAILABLE";
        log.warn("Returning fallback status '{}' for Customer Id: {}", fallbackStatus, id);
        
        return CustomerMapper.mapToDto(customer, fallbackStatus);
    }
}
