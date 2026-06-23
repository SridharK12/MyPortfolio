package com.idapractice.idapractice.dto;

import java.math.BigDecimal;

public class PaymentResponseDTO {

    private Long paymentId;
    private String fromAccount;
    private String toAccount;
    private BigDecimal amount;
    private String status;
    private String remarks;

    public PaymentResponseDTO() {
    }

    public PaymentResponseDTO(Long paymentId, String fromAccount,
                              String toAccount, BigDecimal amount,String status,
                              String remarks) {
        this.paymentId = paymentId;
        this.fromAccount = fromAccount;
        this.toAccount = toAccount;
        this.amount = amount;
        this.status=status;
        this.remarks = remarks;
    }

    public Long getPaymentId() {
        return paymentId;
    }

    public void setPaymentId(Long paymentId) {
        this.paymentId = paymentId;
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

    public void setStatus(String status) {
        this.status = status;
    }
    public String getStatus() {
        return status;
    }

    public void setToAccount(String toAccount) {
        this.toAccount = toAccount;
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