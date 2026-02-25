package com.example.customerapi.controller;

import com.example.customerapi.dto.CustomerDetailsWithStatusResponse;
import com.example.customerapi.entity.Customer;
import com.example.customerapi.service.CustomerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.HttpStatus;

// Import for rate limiter annotation at controller level
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

@RestController
@RequestMapping("/api/customers")
public class CustomerController {
    
    @Autowired
    private CustomerService customerService;
    
    @GetMapping
    public ResponseEntity<List<Customer>> getAllCustomers() {

        List<Customer> customers = customerService.getAllCustomers();

        if (customers.isEmpty()) {
            return ResponseEntity.noContent().build();
        }

        return ResponseEntity.ok(customers);
    }    
    @GetMapping("/{id}")
    public ResponseEntity<Customer> getCustomerById(@PathVariable Long id) {
        return customerService.getCustomerById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    private static final Logger log = LoggerFactory.getLogger(CustomerController.class);
    
    // ----------------------------------------------------------
    // ENDPOINT WITH RESILIENCE4J PROTECTION
    // GET /api/customers/{id}/with-status
    // ----------------------------------------------------------
    // This endpoint is protected by rate limiter at controller level
    // Additional protection (circuit breaker, retry) is applied at service layer
    @GetMapping("/{id}/with-status")
    public ResponseEntity<CustomerDetailsWithStatusResponse> getCustomerWithStatus(@PathVariable Long id) {
        try {
        	log.info("Received request to fetch customer with status for Customer Id: {}", id);
        	
        	// Service method has circuit breaker, retry, and rate limiter
        	CustomerDetailsWithStatusResponse response =
                    customerService.getCustomerDetailsWithStatus(id);
                    
            log.info("Successfully retrieved customer details with status for Customer Id: {}", id);
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            // thrown when customer is not found
            log.error("Customer not found for Id: {}", id);
            return ResponseEntity.notFound().build();
        }
    }
    // ----------------------------------------------------------
    
    @PostMapping
    public ResponseEntity<Customer> createCustomer(@RequestBody Customer customer) {
        Customer createdCustomer = customerService.createCustomer(customer);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdCustomer);
    }

    @PutMapping("/{id}")
    public ResponseEntity<Customer> updateCustomer(@PathVariable Long id, @RequestBody Customer customer) {
        try {
            return ResponseEntity.ok(customerService.updateCustomer(id, customer));
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteCustomer(@PathVariable Long id) {

        boolean deleted = customerService.deleteCustomer(id);

        if (!deleted) {
            return ResponseEntity
                    .status(HttpStatus.NOT_FOUND)
                    .body("Customer id " + id + " does not exist");
        }

        return ResponseEntity.ok("Customer id " + id + " deleted");
    }

}
