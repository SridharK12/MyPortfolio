package com.idapractice.idapractice.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.util.Optional;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.idapractice.idapractice.dto.AuthorizationRequestDTO;
import com.idapractice.idapractice.dto.PaymentDTO;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.dto.PaymentUpdateDTO;
import com.idapractice.idapractice.entity.Payment;
import com.idapractice.idapractice.entity.Account;
import com.idapractice.idapractice.entity.PaymentOutboxEvent;
import com.idapractice.idapractice.enums.PaymentStatus;
import com.idapractice.idapractice.exception.InvalidPaymentOperationException;
import com.idapractice.idapractice.exception.AccountNotFoundException;
import com.idapractice.idapractice.exception.PaymentNotFoundException;
import com.idapractice.idapractice.outbox.PaymentOutboxRepository;
import com.idapractice.idapractice.repository.PaymentRepository;
import com.idapractice.idapractice.repository.AccountRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

/**
 * All payment business logic.
 *
 * Key production patterns applied here:
 *
 * 1. OUTBOX instead of direct Kafka publish
 *    Every state change writes an outbox row in the SAME @Transactional boundary
 *    as the payment mutation. PaymentOutboxPublisher polls and sends to Kafka
 *    separately. This eliminates the dual-write problem.
 *
 * 2. IDEMPOTENCY KEY
 *    If the caller supplies X-Idempotency-Key and that key already exists,
 *    the existing payment is returned without re-inserting. Safe for retries.
 *
 * 3. LIFECYCLE GUARD (guardPendingOnly)
 *    APPROVED / REJECTED / CANCELLED payments are immutable.
 *    Any attempt to mutate them raises InvalidPaymentOperationException → 422.
 *
 * 4. OPTIMISTIC LOCKING
 *    The @Version field on Payment detects concurrent updates.
 *    The losing write throws ObjectOptimisticLockingFailureException → 409.
 *    Handled by GlobalExceptionHandler.
 *
 * Kafka topics (written via outbox):
 *  payment-created   → POST   /v1/payments
 *  payment-updated   → PUT    /v1/payments/{id}
 *  payment-approved  → POST   /v1/payments/{id}/authorization (APPROVED)
 *  payment-rejected  → POST   /v1/payments/{id}/authorization (REJECTED)
 *  payment-cancelled → DELETE /v1/payments/{id}
 */
@Service
@Transactional
public class PaymentService {

    private static final Logger log = LoggerFactory.getLogger(PaymentService.class);

    // ── Kafka topic constants ────────────────────────────────────────────────
    static final String TOPIC_CREATED   = "payment-created";
    static final String TOPIC_UPDATED   = "payment-updated";
    static final String TOPIC_APPROVED  = "payment-approved";
    static final String TOPIC_REJECTED  = "payment-rejected";
    static final String TOPIC_CANCELLED = "payment-cancelled";

    private final PaymentRepository      paymentRepository;
    private final PaymentOutboxRepository outboxRepository;
    private final ObjectMapper           objectMapper;
    private final AccountRepository      accountRepository;

    public PaymentService(PaymentRepository paymentRepository,
                          PaymentOutboxRepository outboxRepository,
                          AccountRepository accountRepository,
                          ObjectMapper objectMapper) {
        this.paymentRepository = paymentRepository;
        this.outboxRepository  = outboxRepository;
        this.objectMapper      = objectMapper;
        this.accountRepository = accountRepository;
    }

    // ── CREATE ───────────────────────────────────────────────────────────────

    /**
     * Creates a new PENDING payment.
     *
     * @param paymentDTO      validated request body (fromAccount, toAccount, amount, remarks)
     * @param idempotencyKey  optional X-Idempotency-Key header; null if not supplied
     */
    public PaymentResponseDTO createPayment(PaymentDTO paymentDTO, String idempotencyKey) {

        // Idempotency check: if this key was seen before, return the cached result
        if (StringUtils.hasText(idempotencyKey)) {
            return paymentRepository.findByIdempotencyKey(idempotencyKey)
                    .map(existing -> {
                        log.info("Idempotent replay for key={} → paymentId={}", idempotencyKey, existing.getPaymentId());
                        return buildResponse(existing);
                    })
                    .orElseGet(() -> persistNewPayment(paymentDTO, idempotencyKey));
        }

        return persistNewPayment(paymentDTO, null);
    }

    private PaymentResponseDTO persistNewPayment(PaymentDTO dto, String idempotencyKey) {

        Payment payment = new Payment();
        Account account = new Account();
        boolean fromAccountExists;
        boolean toAccountExists;
        fromAccountExists=accountRepository.existsByAccountNumber(dto.getFromAccount().trim());
        toAccountExists=accountRepository.existsByAccountNumber(dto.getToAccount().trim());
        Optional <Account> exAccount = accountRepository.findByAccountHolderName("ABCD");
//        					.orElseThrow(() -> new AccountNotFoundException("Account not found"));
        
        if (fromAccountExists)
        	payment.setFromAccount(dto.getFromAccount().trim());
        else
        	throw new AccountNotFoundException("*********PLEASE CHECK FROM ACCOUNT NUMBER********");
        
        if (toAccountExists)
        	payment.setToAccount(dto.getToAccount().trim());
        else
        	throw new AccountNotFoundException("*********PLEASE CHECK   TO ACCOUNT NUMBER********");
        
        payment.setAmount(dto.getAmount());
        payment.setRemarks(dto.getRemarks());
        payment.setStatus(PaymentStatus.PENDING);
        payment.setIdempotencyKey(idempotencyKey);

        Payment saved = paymentRepository.save(payment);
        PaymentResponseDTO response = buildResponse(saved);

        writeOutbox(saved.getPaymentId(), TOPIC_CREATED, response);

        log.info("Payment created: id={} from={} to={} amount={}",
                saved.getPaymentId(), saved.getFromAccount(),
                saved.getToAccount(), saved.getAmount());

        return response;
    }

    // ── READ (single) ────────────────────────────────────────────────────────

    @Transactional(readOnly = true)
    public PaymentResponseDTO getPaymentById(Long paymentId) {
        return buildResponse(findOrThrow(paymentId));
    }

    // ── READ (list with filters + pagination) ────────────────────────────────

    /**
     * @param statusStr    optional status filter (PENDING / APPROVED / REJECTED / CANCELLED)
     * @param fromAccount  optional sender account filter
     * @param page         zero-based page index
     * @param size         page size — capped at 100 to prevent runaway queries
     */
    @Transactional(readOnly = true)
    public Page<PaymentResponseDTO> getAllPayments(
            String statusStr, String fromAccount, int page, int size) {

        int cappedSize = Math.min(size, 100);
        Pageable pageable = PageRequest.of(
                page, cappedSize, Sort.by(Sort.Direction.DESC, "createdAt"));

        PaymentStatus status = null;
        if (StringUtils.hasText(statusStr)) {
            try {
                status = PaymentStatus.valueOf(statusStr.trim().toUpperCase());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        "Invalid status value: '" + statusStr +
                        "'. Must be one of: PENDING, APPROVED, REJECTED, CANCELLED");
            }
        }

        String account = StringUtils.hasText(fromAccount) ? fromAccount.trim() : null;

        return paymentRepository
                .findByFilters(status, account, pageable)
                .map(this::buildResponse);
    }

    // ── UPDATE ───────────────────────────────────────────────────────────────

    /**
     * Updates mutable fields on a PENDING payment (partial-update semantics).
     * Null fields in the DTO mean "no change".
     *
     * @throws PaymentNotFoundException         if payment does not exist
     * @throws InvalidPaymentOperationException if payment is not PENDING
     */
    public PaymentResponseDTO updatePayment(Long paymentId, PaymentUpdateDTO dto) {

        Payment payment = findOrThrow(paymentId);
        guardPendingOnly(payment, "update");

        if (StringUtils.hasText(dto.getFromAccount())) {
            payment.setFromAccount(dto.getFromAccount().trim());
        }
        if (StringUtils.hasText(dto.getToAccount())) {
            payment.setToAccount(dto.getToAccount().trim());
        }
        if (dto.getAmount() != null) {
            payment.setAmount(dto.getAmount());
        }
        if (dto.getRemarks() != null) {        // blank remarks are valid (clearing the field)
            payment.setRemarks(dto.getRemarks());
        }

        Payment saved = paymentRepository.save(payment);
        PaymentResponseDTO response = buildResponse(saved);

        writeOutbox(saved.getPaymentId(), TOPIC_UPDATED, response);

        log.info("Payment updated: id={}", saved.getPaymentId());
        return response;
    }

    // ── AUTHORIZE ────────────────────────────────────────────────────────────

    /**
     * Approves or rejects a PENDING payment.
     *
     * @throws PaymentNotFoundException         if payment does not exist
     * @throws InvalidPaymentOperationException if payment is not PENDING
     */
    public PaymentResponseDTO authorizePayment(Long paymentId, AuthorizationRequestDTO req) {

        Payment payment = findOrThrow(paymentId);
        guardPendingOnly(payment, "authorize");

        PaymentStatus decision = PaymentStatus.valueOf(req.getStatus().trim().toUpperCase());
        payment.setStatus(decision);

        if (StringUtils.hasText(req.getRemarks())) {
            payment.setRemarks(req.getRemarks());
        }

        Payment saved = paymentRepository.save(payment);
        PaymentResponseDTO response = buildResponse(saved);

        String topic = (decision == PaymentStatus.APPROVED) ? TOPIC_APPROVED : TOPIC_REJECTED;
        writeOutbox(saved.getPaymentId(), topic, response);

        log.info("Payment {}d: id={}", decision.name().toLowerCase(), saved.getPaymentId());
        return response;
    }

    // ── DELETE (soft) ────────────────────────────────────────────────────────

    /**
     * Soft-deletes a PENDING payment by moving it to CANCELLED.
     *
     * Hard delete is avoided to:
     *  - Preserve the full audit trail in the DB
     *  - Allow downstream Kafka consumers to react to the cancellation
     *
     * @throws PaymentNotFoundException         if payment does not exist
     * @throws InvalidPaymentOperationException if payment is not PENDING
     */
    public void cancelPayment(Long paymentId) {

        Payment payment = findOrThrow(paymentId);
        guardPendingOnly(payment, "cancel");

        payment.setStatus(PaymentStatus.CANCELLED);
        Payment saved = paymentRepository.save(payment);

        writeOutbox(saved.getPaymentId(), TOPIC_CANCELLED, buildResponse(saved));

        log.info("Payment cancelled: id={}", saved.getPaymentId());
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private Payment findOrThrow(Long paymentId) {
        return paymentRepository.findById(paymentId)
                .orElseThrow(() -> new PaymentNotFoundException(paymentId));
    }

    /**
     * Enforces that only PENDING payments can undergo state-changing operations.
     * APPROVED, REJECTED and CANCELLED are terminal (immutable).
     */
    private void guardPendingOnly(Payment payment, String operation) {
        if (payment.getStatus() != PaymentStatus.PENDING) {
            throw new InvalidPaymentOperationException(
                    String.format(
                        "Cannot %s payment %d — current status is '%s'. " +
                        "Only PENDING payments may be %sd.",
                        operation, payment.getPaymentId(),
                        payment.getStatus().name(), operation));
        }
    }

    /**
     * Writes a Kafka event to the outbox table within the current transaction.
     * The outbox row is committed atomically with the payment row.
     * PaymentOutboxPublisher picks it up asynchronously and sends to Kafka.
     */
    private void writeOutbox(Long paymentId, String topic, PaymentResponseDTO response) {
        try {
            String json = objectMapper.writeValueAsString(response);

            PaymentOutboxEvent event = new PaymentOutboxEvent();
            event.setPaymentId(paymentId);
            event.setTopic(topic);
            event.setMessageKey(String.valueOf(paymentId));
            event.setPayload(json);

            outboxRepository.save(event);

        } catch (JsonProcessingException e) {
            // This should never happen with a well-formed DTO.
            // If it does, fail the entire transaction so neither the payment
            // nor the outbox row are committed (consistent failure).
            throw new RuntimeException("Failed to serialise outbox event for paymentId=" + paymentId, e);
        }
    }

    private PaymentResponseDTO buildResponse(Payment p) {
        PaymentResponseDTO dto = new PaymentResponseDTO();
        dto.setPaymentId(p.getPaymentId());
        dto.setFromAccount(p.getFromAccount());
        dto.setToAccount(p.getToAccount());
        dto.setAmount(p.getAmount());
        dto.setRemarks(p.getRemarks());
        dto.setStatus(p.getStatus());
        dto.setVersion(p.getVersion());
        dto.setCreatedAt(p.getCreatedAt());
        dto.setUpdatedAt(p.getUpdatedAt());
        return dto;
    }
}
