package com.example.customerapi.dto;

import java.time.LocalDate;

public class CustomerDetailsWithStatusResponse {

    private Long customerId;
    private String customerName;
    private LocalDate customerDob;

    private String customerSsn;      // JIRA-1
    private String customerState;    // JIRA-1

    private String modifiedBy;
    private LocalDate modificationDate;
    private String status;

    public CustomerDetailsWithStatusResponse() {
    }

    public CustomerDetailsWithStatusResponse(Long customerId,
                                             String customerName,
                                             LocalDate customerDob,
                                             String modifiedBy,
                                             LocalDate modificationDate,
                                             String status) {
        this.customerId = customerId;
        this.customerName = customerName;
        this.customerDob = customerDob;
        this.modifiedBy = modifiedBy;
        this.modificationDate = modificationDate;
        this.status = status;
    }

    public Long getCustomerId() {
        return customerId;
    }

    public void setCustomerId(Long customerId) {
        this.customerId = customerId;
    }

    public String getCustomerName() {
        return customerName;
    }

    public void setCustomerName(String customerName) {
        this.customerName = customerName;
    }

    public LocalDate getCustomerDob() {
        return customerDob;
    }

    public void setCustomerDob(LocalDate customerDob) {
        this.customerDob = customerDob;
    }

    public String getCustomerSsn() {          // JIRA-1
        return customerSsn;
    }

    public void setCustomerSsn(String customerSsn) {   // JIRA-1
        this.customerSsn = customerSsn;
    }

    public String getCustomerState() {        // JIRA-1
        return customerState;
    }

    public void setCustomerState(String customerState) {   // JIRA-1
        this.customerState = customerState;
    }

    public String getModifiedBy() {
        return modifiedBy;
    }

    public void setModifiedBy(String modifiedBy) {
        this.modifiedBy = modifiedBy;
    }

    public LocalDate getModificationDate() {
        return modificationDate;
    }

    public void setModificationDate(LocalDate modificationDate) {
        this.modificationDate = modificationDate;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
}
