package com.idapractice.idapractice.dto;

import java.math.BigDecimal;

public class PaymentDTO {

    private String fromAccount;
    private String toAccount;
    private BigDecimal amount;
    private String remarks;
    private String status;

    public PaymentDTO() {
    }

    public PaymentDTO(String fromAccount, String toAccount,
                      BigDecimal amount, String status, String remarks) {
        this.fromAccount = fromAccount;
        this.toAccount = toAccount;
        this.amount = amount;
        this.status=status;
        this.remarks = remarks;
    }

    public String getFromAccount() {
        return fromAccount;
    }

    public void setFromAccount(String fromAccount) {
        this.fromAccount = fromAccount;
    }

    public String getToAccount() {
        return toAccount;
    }

    public void setToAccount(String toAccount) {
        this.toAccount = toAccount;
    }
    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
        this.amount = amount;
    }

    public String getRemarks() {
        return remarks;
    }

    public void setRemarks(String remarks) {
        this.remarks = remarks;
    }
}