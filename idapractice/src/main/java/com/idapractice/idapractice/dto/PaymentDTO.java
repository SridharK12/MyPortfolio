package com.idapractice.idapractice.dto;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Digits;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.math.BigDecimal;

public class PaymentDTO {

    @NotBlank(message = "fromAccount is mandatory")
    @Size(max = 100, message = "fromAccount must not exceed 100 characters")
    private String fromAccount;

    @NotBlank(message = "toAccount is mandatory")
    @Size(max = 100, message = "toAccount must not exceed 100 characters")
    private String toAccount;

    @NotNull(message = "amount is mandatory")
    @DecimalMin(value = "0.01", message = "amount must be greater than zero")
    @Digits(integer = 15, fraction = 4, message = "amount exceeds allowed precision (15 integer, 4 decimal)")
    private BigDecimal amount;

    @Size(max = 500, message = "remarks must not exceed 500 characters")
    private String remarks;

    public PaymentDTO() {}

    public String getFromAccount()          { return fromAccount; }
    public void setFromAccount(String v)    { this.fromAccount = v; }

    public String getToAccount()            { return toAccount; }
    public void setToAccount(String v)      { this.toAccount = v; }

    public BigDecimal getAmount()           { return amount; }
    public void setAmount(BigDecimal v)     { this.amount = v; }

    public String getRemarks()              { return remarks; }
    public void setRemarks(String v)        { this.remarks = v; }
}
