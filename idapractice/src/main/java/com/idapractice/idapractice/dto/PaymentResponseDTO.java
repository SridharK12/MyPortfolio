package com.idapractice.idapractice.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.idapractice.idapractice.enums.PaymentStatus;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class PaymentResponseDTO {

    private Long paymentId;
    private String fromAccount;
    private String toAccount;
    private BigDecimal amount;
    private String remarks;
    private PaymentStatus status;
    private Long version;

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createdAt;

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime updatedAt;

    public PaymentResponseDTO() {}

    // ── Getters & setters ────────────────────────────────────────────────────

    public Long getPaymentId()                  { return paymentId; }
    public void setPaymentId(Long v)            { this.paymentId = v; }

    public String getFromAccount()              { return fromAccount; }
    public void setFromAccount(String v)        { this.fromAccount = v; }

    public String getToAccount()                { return toAccount; }
    public void setToAccount(String v)          { this.toAccount = v; }

    public BigDecimal getAmount()               { return amount; }
    public void setAmount(BigDecimal v)         { this.amount = v; }

    public String getRemarks()                  { return remarks; }
    public void setRemarks(String v)            { this.remarks = v; }

    public PaymentStatus getStatus()            { return status; }
    public void setStatus(PaymentStatus v)      { this.status = v; }

    public Long getVersion()                    { return version; }
    public void setVersion(Long v)              { this.version = v; }

    public LocalDateTime getCreatedAt()         { return createdAt; }
    public void setCreatedAt(LocalDateTime v)   { this.createdAt = v; }

    public LocalDateTime getUpdatedAt()         { return updatedAt; }
    public void setUpdatedAt(LocalDateTime v)   { this.updatedAt = v; }
}
