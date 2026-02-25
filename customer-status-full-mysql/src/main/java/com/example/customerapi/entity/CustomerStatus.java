package com.example.customerapi.entity;

import javax.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "customer_status")
public class CustomerStatus {

    @Id
    @Column(name = "customer_id")
    private Long customerId;

    @Column(name = "status", nullable = false)
    private String status;

    @Column(name = "modified_by")
    private String modifiedBy;

    @Column(name = "modification_date")
    private LocalDateTime modificationDate;
    
    public Long getCustomerId() {
        return customerId;
    }

    public String getStatus() {
        return status;
    }

    public String getModifiedBy() {
        return modifiedBy;
    }

    public LocalDateTime getModificationDate() {
        return modificationDate;
    }
    public void setCustomerId(Long customerId) {
        this.customerId = customerId;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setModifiedBy(String modifiedBy) {
        this.modifiedBy = modifiedBy;
    }

    public void setModificationDate(LocalDateTime modificationDate) {
        this.modificationDate = modificationDate;
    }
}
