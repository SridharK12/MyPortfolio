package com.idapractice.idapractice.dto;

import jakarta.validation.constraints.*;
import com.idapractice.idapractice.enums.AccountType;
import java.math.BigDecimal;

public class AccountDTO {

    private Long id;

    @NotBlank(message = "Account number is required")
    @Size(min = 8, max = 16, message = "Account number must be 8-16 characters")
    private String accountNumber;

    @NotBlank(message = "Account holder name is required")
    @Size(min = 2, max = 100, message = "Name must be 2-100 characters")
    private String accountHolderName;

    @NotNull(message = "Account type is required")
    private AccountType accountType;

    @NotNull(message = "Balance is required")
    @DecimalMin(value = "0.0", message = "Balance cannot be negative")
    private BigDecimal balance;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getAccountNumber() { return accountNumber; }
    public void setAccountNumber(String accountNumber) { this.accountNumber = accountNumber; }
    public String getAccountHolderName() { return accountHolderName; }
    public void setAccountHolderName(String accountHolderName) { this.accountHolderName = accountHolderName; }
    public AccountType getAccountType() { return accountType; }
    public void setAccountType(AccountType accountType) { this.accountType = accountType; }
    public BigDecimal getBalance() { return balance; }
    public void setBalance(BigDecimal balance) { this.balance = balance; }
}