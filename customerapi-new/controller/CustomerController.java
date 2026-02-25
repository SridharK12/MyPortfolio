package com.example.customerapi.controller;

import com.example.customerapi.dto.CustomerDetailsWithStatusResponse;
import com.example.customerapi.entity.Customer;
import com.example.customerapi.service.CustomerService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.HttpStatus;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

@RestController
@RequestMapping("/api/customers")
public class CustomerController {

    @Autowired
    private CustomerService customerService;

    private static final Logger log =
            LoggerFactory.getLogger(CustomerController.class);

    // ----------------------------------------------------------
    // GET ALL OR GET BY NAME (RequestParam)
    // GET /api/customers
    // GET /api/customers?name=John
    // ----------------------------------------------------------
    @GetMapping
    public ResponseEntity<List<Customer>> getCustomers(
            @RequestParam(name = "name", required = false) String customerName) {

        if (customerName != null && !customerName.isBlank()) {
            return ResponseEntity.ok(
                    customerService.getCustomersByName(customerName)
            );
        }

        return ResponseEntity.ok(customerService.getAllCustomers());
    }

    // ----------------------------------------------------------
    // GET BY ID (PathVariable)
    // ----------------------------------------------------------
    @GetMapping("/{id}")
    public ResponseEntity<Customer> getCustomerById(@PathVariable Long id) {

        return customerService.getCustomerById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    // ----------------------------------------------------------
    // GET CUSTOMER WITH STATUS (Resilience4j protected)
    // ----------------------------------------------------------
    @GetMapping("/{id}/with-status")
    public ResponseEntity<CustomerDetailsWithStatusResponse>
    getCustomerWithStatus(@PathVariable Long id) {

        try {
            log.info(
                "Received request to fetch customer with status for Customer Id: {}",
                id
            );

            CustomerDetailsWithStatusResponse response =
                    customerService.getCustomerDetailsWithStatus(id);

            return ResponseEntity.ok(response);

        } catch (RuntimeException e) {
            log.error("Customer not found for Id: {}", id);
            return ResponseEntity.notFound().build();
        }
    }

    // ----------------------------------------------------------
    // CREATE CUSTOMER
    // ----------------------------------------------------------
    @PostMapping
    public ResponseEntity<Customer> createCustomer(
            @RequestBody Customer customer) {

        Customer createdCustomer =
                customerService.createCustomer(customer);

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(createdCustomer);
    }

    // ----------------------------------------------------------
    // UPDATE CUSTOMER
    // ----------------------------------------------------------
    @PutMapping("/{id}")
    public ResponseEntity<Customer> updateCustomer(
            @PathVariable Long id,
            @RequestBody Customer customer) {

        try {
            return ResponseEntity.ok(
                    customerService.updateCustomer(id, customer)
            );
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }

    // ----------------------------------------------------------
    // DELETE CUSTOMER
    // ----------------------------------------------------------
    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteCustomer(
            @PathVariable Long id) {

        boolean deleted = customerService.deleteCustomer(id);

        if (!deleted) {
            return ResponseEntity
                    .status(HttpStatus.NOT_FOUND)
                    .body("Customer id " + id + " does not exist");
        }

        return ResponseEntity.ok(
                "Customer id " + id + " deleted"
        );
    }
}
