package com.idapractice.idapractice.dto;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Digits;
import jakarta.validation.constraints.Size;

import java.math.BigDecimal;

/**
 * Request body for PUT /v1/payments/{id}.
 *
 * Null fields are treated as "no change" (partial-update semantics).
 * The caller cannot change status through this DTO — that is exclusively
 * controlled by the /authorization endpoint to enforce maker-checker separation.
 */
public class PaymentUpdateDTO {

    @Size(max = 100, message = "fromAccount must not exceed 100 characters")
    private String fromAccount;

    @Size(max = 100, message = "toAccount must not exceed 100 characters")
    private String toAccount;

    @DecimalMin(value = "0.01", message = "amount must be greater than zero")
    @Digits(integer = 15, fraction = 4, message = "amount exceeds allowed precision")
    private BigDecimal amount;

    @Size(max = 500, message = "remarks must not exceed 500 characters")
    private String remarks;

    public PaymentUpdateDTO() {}

    public String getFromAccount()          { return fromAccount; }
    public void setFromAccount(String v)    { this.fromAccount = v; }

    public String getToAccount()            { return toAccount; }
    public void setToAccount(String v)      { this.toAccount = v; }

    public BigDecimal getAmount()           { return amount; }
    public void setAmount(BigDecimal v)     { this.amount = v; }

    public String getRemarks()              { return remarks; }
    public void setRemarks(String v)        { this.remarks = v; }
}
