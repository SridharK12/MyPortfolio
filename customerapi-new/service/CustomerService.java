package com.example.customerapi.service;

import com.example.customerapi.dto.CustomerDetailsWithStatusResponse;
import com.example.customerapi.dto.CustomerStatusResponse;
import com.example.customerapi.entity.Customer;
import com.example.customerapi.entity.CustomerAdditionalDetails;
import com.example.customerapi.repository.CustomerRepository;
import com.example.customerapi.mapper.CustomerMapper;
import com.example.customerapi.exception.BusinessException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.util.StringUtils;

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
    private RestTemplate restTemplate;

    private static final String STATUS_URL =
            "http://customer-status-service/api/customers/{id}";

    private static final Logger log =
            LoggerFactory.getLogger(CustomerService.class);

    // -------------------------
    // READ OPERATIONS
    // -------------------------

    public List<Customer> getAllCustomers() {
        return customerRepository.findAll();
    }

    public Optional<Customer> getCustomerById(Long id) {
        return customerRepository.findById(id);
    }

    // -------------------------
    // CREATE CUSTOMER
    // -------------------------

    public Customer createCustomer(Customer customer) {

        if (!StringUtils.hasText(customer.getCustomerState())) {
            throw new BusinessException(
                    "STATE_CODE_MISSING",
                    "State code must not be null or empty",
                    HttpStatus.BAD_REQUEST
            );
        }

        customer.setModificationDate(LocalDate.now());

        // 🔑 Aggregate ownership enforcement
        if (customer.getAdditionalDetails() != null) {
            customer.getAdditionalDetails().setCustomer(customer);
        }

        return customerRepository.save(customer);
    }

    // -------------------------
    // UPDATE CUSTOMER
    // -------------------------

    public Customer updateCustomer(Long id, Customer customerDetails) {

        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Customer not found"));

        log.info("Updating Customer Record for Customer Id {}", id);

        customer.setCustomerName(customerDetails.getCustomerName());
        customer.setCustomerDob(customerDetails.getCustomerDob());
        customer.setModifiedBy(customerDetails.getModifiedBy());
        customer.setModificationDate(LocalDate.now());

        // 🔑 Handle CustomerAdditionalDetails lifecycle
        if (customerDetails.getAdditionalDetails() != null) {

            if (customer.getAdditionalDetails() == null) {
                // Create new additional details
                CustomerAdditionalDetails addl =
                        customerDetails.getAdditionalDetails();
                addl.setCustomer(customer);
                customer.setAdditionalDetails(addl);

            } else {
                // Update existing additional details
                customer.getAdditionalDetails().setAddlDtl1(
                        customerDetails.getAdditionalDetails().getAddlDtl1()
                );
                customer.getAdditionalDetails().setAddlDtl2(
                        customerDetails.getAdditionalDetails().getAddlDtl2()
                );
                customer.getAdditionalDetails().setAddlDtl3(
                        customerDetails.getAdditionalDetails().getAddlDtl3()
                );
            }

        } else {
            // Remove additional details (orphanRemoval = true)
            customer.setAdditionalDetails(null);
        }

        return customerRepository.save(customer);
    }

    // -------------------------
    // DELETE CUSTOMER
    // -------------------------

    public boolean deleteCustomer(Long id) {

        if (!customerRepository.existsById(id)) {
            return false;
        }

        // Cascades delete to CustomerAdditionalDetails
        customerRepository.deleteById(id);
        return true;
    }

    // ---------------------------------------------------------
    // CUSTOMER + STATUS WITH RESILIENCE4J
    // ---------------------------------------------------------

    @CircuitBreaker(
            name = "customerStatusCircuitBreaker",
            fallbackMethod = "getCustomerDetailsWithStatusFallback"
    )
    @Retry(name = "customerStatusRetry")
    @RateLimiter(name = "customerStatusRateLimiter")
    public CustomerDetailsWithStatusResponse
    getCustomerDetailsWithStatus(Long id) {

        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Customer not found"));

        log.info("Before calling CustomerStatus API for Customer Id: {}", id);

        CustomerStatusResponse statusResponse =
                restTemplate.getForObject(
                        STATUS_URL,
                        CustomerStatusResponse.class,
                        id
                );

        String status =
                (statusResponse != null)
                        ? statusResponse.getStatus()
                        : "UNKNOWN";

        log.info("Received status from CustomerStatus API: {}", status);

        return CustomerMapper.mapToDto(customer, status);
    }

    // ---------------------------------------------------------
    // FALLBACK METHOD
    // ---------------------------------------------------------

    public CustomerDetailsWithStatusResponse
    getCustomerDetailsWithStatusFallback(Long id, Throwable throwable) {

        log.error(
                "Fallback method called for Customer Id {}. Reason: {}",
                id,
                throwable.getMessage()
        );

        Customer customer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Customer not found"));

        String fallbackStatus = "UNAVAILABLE";

        return CustomerMapper.mapToDto(customer, fallbackStatus);
    }
}
