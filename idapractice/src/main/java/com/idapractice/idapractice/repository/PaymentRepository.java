package com.idapractice.idapractice.repository;

import com.idapractice.idapractice.entity.Payment;
import com.idapractice.idapractice.enums.PaymentStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface PaymentRepository extends JpaRepository<Payment, Long> {

    // Idempotency check — returns the existing payment if the key was already used
    Optional<Payment> findByIdempotencyKey(String idempotencyKey);

    /**
     * Flexible paginated query for GET /v1/payments.
     *
     * Both params are optional — pass null to skip each filter.
     * The IS NULL check in JPQL lets a single query cover all four
     * combinations (no filter, status only, fromAccount only, both).
     *
     * Sorted by the Pageable passed in (default: createdAt DESC).
     */
    @Query("SELECT p FROM Payment p " +
           "WHERE (:status IS NULL      OR p.status      = :status) " +
           "AND   (:fromAccount IS NULL OR p.fromAccount = :fromAccount)")
    Page<Payment> findByFilters(
            @Param("status")      PaymentStatus status,
            @Param("fromAccount") String fromAccount,
            Pageable pageable);
}
