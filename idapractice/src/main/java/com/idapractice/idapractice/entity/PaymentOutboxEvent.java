package com.idapractice.idapractice.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

/**
 * Transactional Outbox table.
 *
 * Instead of calling kafkaTemplate.send() directly inside a @Transactional
 * service method (which creates a dual-write problem between MySQL and Kafka),
 * we write the event payload into this table in the SAME DB transaction as the
 * payment mutation.
 *
 * A separate @Scheduled publisher (PaymentOutboxPublisher) polls for
 * published=false rows, sends them to Kafka, and marks them published=true.
 *
 * Guarantees:
 *  - If the DB transaction rolls back, the outbox row is also rolled back
 *    → no phantom Kafka messages for payments that don't exist.
 *  - If Kafka is temporarily down, rows remain in the outbox and are retried
 *    on the next poll cycle → no silent message loss.
 */
@Entity
@Table(name = "payment_outbox")
public class PaymentOutboxEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long paymentId;

    @Column(nullable = false, length = 100)
    private String topic;

    /** Kafka message key (String.valueOf(paymentId)) — ensures ordering per payment. */
    @Column(nullable = false, length = 50)
    private String messageKey;

    /** Full JSON of PaymentResponseDTO, serialized at write time. */
    @Column(nullable = false, columnDefinition = "TEXT")
    private String payload;

    @Column(nullable = false)
    private boolean published = false;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdAt;

    private LocalDateTime publishedAt;

    public PaymentOutboxEvent() {}

    // ── Getters & setters ────────────────────────────────────────────────────

    public Long getId()                         { return id; }
    public void setId(Long id)                  { this.id = id; }

    public Long getPaymentId()                  { return paymentId; }
    public void setPaymentId(Long v)            { this.paymentId = v; }

    public String getTopic()                    { return topic; }
    public void setTopic(String v)              { this.topic = v; }

    public String getMessageKey()               { return messageKey; }
    public void setMessageKey(String v)         { this.messageKey = v; }

    public String getPayload()                  { return payload; }
    public void setPayload(String v)            { this.payload = v; }

    public boolean isPublished()                { return published; }
    public void setPublished(boolean v)         { this.published = v; }

    public LocalDateTime getCreatedAt()         { return createdAt; }
    public void setCreatedAt(LocalDateTime v)   { this.createdAt = v; }

    public LocalDateTime getPublishedAt()       { return publishedAt; }
    public void setPublishedAt(LocalDateTime v) { this.publishedAt = v; }
}
