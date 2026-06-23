package com.idapractice.idapractice.entity;

import com.idapractice.idapractice.enums.PaymentStatus;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "payments")
public class Payment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long paymentId;

    @Column(nullable = false, length = 100)
    private String fromAccount;

    @Column(nullable = false, length = 100)
    private String toAccount;

    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal amount;

    @Column(length = 500)
    private String remarks;

    /**
     * Stored as VARCHAR so the value is human-readable directly in the DB.
     * Using EnumType.STRING (not ORDINAL) means adding new enum values
     * never shifts existing rows.
     */
    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private PaymentStatus status;

    /**
     * Optional caller-supplied deduplication token.
     * When provided, the service returns the existing payment instead of
     * creating a duplicate — safe for client retries.
     */
    @Column(unique = true, length = 100)
    private String idempotencyKey;

    /**
     * Optimistic locking — prevents lost updates when two requests
     * concurrently read and attempt to modify the same payment.
     * JPA increments this on every UPDATE; a stale write raises
     * ObjectOptimisticLockingFailureException → mapped to HTTP 409.
     */
    @Version
    @Column(nullable = false)
    private Long version;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    public Payment() {}

    // ── Getters & setters ────────────────────────────────────────────────────

    public Long getPaymentId()                  { return paymentId; }
    public void setPaymentId(Long paymentId)    { this.paymentId = paymentId; }

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

    public String getIdempotencyKey()           { return idempotencyKey; }
    public void setIdempotencyKey(String v)     { this.idempotencyKey = v; }

    public Long getVersion()                    { return version; }
    public void setVersion(Long v)              { this.version = v; }

    public LocalDateTime getCreatedAt()         { return createdAt; }
    public void setCreatedAt(LocalDateTime v)   { this.createdAt = v; }

    public LocalDateTime getUpdatedAt()         { return updatedAt; }
    public void setUpdatedAt(LocalDateTime v)   { this.updatedAt = v; }
}
