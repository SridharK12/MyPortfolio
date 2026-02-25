package com.example.customerapi.dto;

public class CustomerStatusResponse {

    private Long customerId;
    private String status;

    public CustomerStatusResponse() {
    }

    public CustomerStatusResponse(Long customerId, String status) {
        this.customerId = customerId;
        this.status = status;
    }

    public Long getCustomerId() {
        return customerId;
    }

    public void setCustomerId(Long customerId) {
        this.customerId = customerId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
}
