package com.example.customerapi.exception;

public class CustomerStatusNotFoundException extends RuntimeException {
    public CustomerStatusNotFoundException(Long customerId) {
        super("Customer status not found for customerId: " + customerId);
    }
}
