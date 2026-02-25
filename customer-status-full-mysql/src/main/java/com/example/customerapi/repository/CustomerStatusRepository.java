package com.example.customerapi.repository;

import com.example.customerapi.entity.CustomerStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface CustomerStatusRepository extends JpaRepository<CustomerStatus, Long> {
    Optional<CustomerStatus> findByCustomerId(Long customerId);
}
