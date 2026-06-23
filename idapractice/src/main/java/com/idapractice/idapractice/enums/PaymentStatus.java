package com.idapractice.idapractice.enums;

/**
 * Represents every state a payment can be in during its lifecycle.
 *
 * Valid transitions:
 *   PENDING  → APPROVED   (via /authorization endpoint)
 *   PENDING  → REJECTED   (via /authorization endpoint)
 *   PENDING  → CANCELLED  (via DELETE endpoint — soft delete)
 *
 * APPROVED, REJECTED, CANCELLED are terminal states.
 * No further state changes are permitted once reached.
 *
 * Using an enum (instead of a raw String) means invalid statuses
 * are impossible to represent — caught at compile time, not runtime.
 */
public enum PaymentStatus {
    PENDING,
    APPROVED,
    REJECTED,
    CANCELLED
}
