package com.example.customerapi.dto;


public class CustomerStatusRequest {

    private Long customerId;

    public CustomerStatusRequest() {
    }

    public CustomerStatusRequest(Long customerId) {
        this.customerId = customerId;
    }

    public Long getCustomerId() {
        return customerId;
    }

    public void setCustomerId(Long customerId) {
        this.customerId = customerId;
    }
}
