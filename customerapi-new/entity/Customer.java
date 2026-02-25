package com.example.customerapi.entity;

import jakarta.persistence.*;
import java.time.LocalDate;

@Entity
@Table(name = "customer")
public class Customer {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long customerId;
    
    private String customerName;
    
    private LocalDate customerDob;

    // JIRA-1: added customer SSN
    private String customerSsn;

    // JIRA-1: added customer state
    private String customerState;
    
    private String modifiedBy;
    
    private LocalDate modificationDate;

    /**
     * Optional additional details.
     * Lifecycle is owned by Customer.
     * Deleted automatically when Customer is deleted.
     */
    @OneToOne(
        mappedBy = "customer",
        cascade = CascadeType.ALL,
        orphanRemoval = true,
        fetch = FetchType.LAZY,
        optional = true
    )
    private CustomerAdditionalDetails additionalDetails;

    public Customer() {}

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

    // JIRA-1
    public String getCustomerSsn() {
        return customerSsn;
    }

    // JIRA-1
    public void setCustomerSsn(String customerSsn) {
        this.customerSsn = customerSsn;
    }

    // JIRA-1
    public String getCustomerState() {
        return customerState;
    }

    // JIRA-1
    public void setCustomerState(String customerState) {
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

    public CustomerAdditionalDetails getAdditionalDetails() {
        return additionalDetails;
    }

    public void setAdditionalDetails(CustomerAdditionalDetails additionalDetails) {
        this.additionalDetails = additionalDetails;

        // Maintain bidirectional relationship
        if (additionalDetails != null) {
            additionalDetails.setCustomer(this);
        }
    }
}
