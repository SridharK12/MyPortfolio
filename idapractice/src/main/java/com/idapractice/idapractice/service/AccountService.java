package com.idapractice.idapractice.service;

import com.idapractice.idapractice.dto.AccountDTO;
import com.idapractice.idapractice.entity.Account;
import com.idapractice.idapractice.repository.AccountRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AccountService {

    private final AccountRepository accountRepository;

    public AccountService(AccountRepository accountRepository) {
        this.accountRepository = accountRepository;
    }

    public AccountDTO create(AccountDTO dto) {
        if (accountRepository.existsByAccountNumber(dto.getAccountNumber())) {
            throw new RuntimeException("Account number already exists");
        }
        Account account = toEntity(dto);
        return toDTO(accountRepository.save(account));
    }

    public List<AccountDTO> getAll() {
        return accountRepository.findAll()
                .stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    public AccountDTO getById(Long id) {
        Account account = accountRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Account not found: " + id));
        return toDTO(account);
    }

    public AccountDTO update(Long id, AccountDTO dto) {
        Account account = accountRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Account not found: " + id));
        account.setAccountHolderName(dto.getAccountHolderName());
        account.setAccountType(dto.getAccountType());
        account.setBalance(dto.getBalance());
        return toDTO(accountRepository.save(account));
    }

    public void delete(Long id) {
        accountRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Account not found: " + id));
        accountRepository.deleteById(id);
    }

    private Account toEntity(AccountDTO dto) {
        Account account = new Account();
        account.setAccountNumber(dto.getAccountNumber());
        account.setAccountHolderName(dto.getAccountHolderName());
        account.setAccountType(dto.getAccountType());
        account.setBalance(dto.getBalance());
        return account;
    }

    private AccountDTO toDTO(Account account) {
        AccountDTO dto = new AccountDTO();
        dto.setId(account.getId());
        dto.setAccountNumber(account.getAccountNumber());
        dto.setAccountHolderName(account.getAccountHolderName());
        dto.setAccountType(account.getAccountType());
        dto.setBalance(account.getBalance());
        return dto;
    }
}