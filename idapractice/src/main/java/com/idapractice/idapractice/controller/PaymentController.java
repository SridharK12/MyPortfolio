package com.idapractice.idapractice.controller;

import com.idapractice.idapractice.dto.AuthorizationRequestDTO;
import com.idapractice.idapractice.dto.PaymentDTO;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.dto.PaymentUpdateDTO;
import com.idapractice.idapractice.service.PaymentService;
import jakarta.validation.Valid;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST controller for the Payment resource.
 *
 * Endpoint summary:
 *
 *  POST   /v1/payments                       Create PENDING payment      → 201 Created
 *  GET    /v1/payments?status=&fromAccount=  List payments (paginated)   → 200 OK
 *  GET    /v1/payments/{id}                  Get payment by ID           → 200 OK
 *  PUT    /v1/payments/{id}                  Update PENDING payment      → 200 OK
 *  DELETE /v1/payments/{id}                  Soft-cancel PENDING payment → 204 No Content
 *  POST   /v1/payments/{id}/authorization    Approve or reject           → 200 OK
 *
 * The controller is deliberately thin:
 *  - All business rules live in PaymentService
 *  - All error handling lives in GlobalExceptionHandler
 *  - @Valid delegates constraint checking to the validation framework
 */
@RestController
@RequestMapping("/v1/payments")
public class PaymentController {

    private final PaymentService paymentService;

    public PaymentController(PaymentService paymentService) {
        this.paymentService = paymentService;
    }

    // ── CREATE ───────────────────────────────────────────────────────────────

    /**
     * POST /v1/payments
     *
     * Headers:
     *   X-Idempotency-Key (optional) — supply a UUID to make the call retry-safe.
     *   If the same key is submitted twice, the original payment is returned.
     *
     * Returns 201 Created (not 200) — semantically correct for resource creation.
     */
    @PostMapping
    public ResponseEntity<PaymentResponseDTO> createPayment(
            @RequestHeader(value = "X-Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody PaymentDTO paymentDTO) {

        PaymentResponseDTO response = paymentService.createPayment(paymentDTO, idempotencyKey);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    // ── READ (list) ──────────────────────────────────────────────────────────

    /**
     * GET /v1/payments
     *
     * Optional query params:
     *   ?status=PENDING          filter by status (PENDING|APPROVED|REJECTED|CANCELLED)
     *   ?fromAccount=ACC001      filter by sender account
     *   ?page=0&size=20          pagination (size capped at 100 in service layer)
     *
     * Returns a Spring Page envelope:
     *   { "content": [...], "totalElements": N, "totalPages": N, "number": 0 }
     */
    @GetMapping
    public ResponseEntity<Page<PaymentResponseDTO>> getAllPayments(
            @RequestParam(required = false)           String status,
            @RequestParam(required = false)           String fromAccount,
            @RequestParam(defaultValue = "0")         int page,
            @RequestParam(defaultValue = "20")        int size) {

        return ResponseEntity.ok(
                paymentService.getAllPayments(status, fromAccount, page, size));
    }

    // ── READ (single) ────────────────────────────────────────────────────────

    /**
     * GET /v1/payments/{id}
     * Returns 404 via GlobalExceptionHandler if the payment does not exist.
     */
    @GetMapping("/{paymentId}")
    public ResponseEntity<PaymentResponseDTO> getPaymentById(
            @PathVariable Long paymentId) {

        return ResponseEntity.ok(paymentService.getPaymentById(paymentId));
    }

    // ── UPDATE ───────────────────────────────────────────────────────────────

    /**
     * PUT /v1/payments/{id}
     *
     * Partial-update semantics: null fields are ignored (no change).
     * Only PENDING payments are editable — returns 422 otherwise.
     *
     * Example:
     *   { "amount": 7500.00, "remarks": "Corrected amount" }
     *   → changes only amount and remarks; fromAccount/toAccount unchanged.
     */
    @PutMapping("/{paymentId}")
    public ResponseEntity<PaymentResponseDTO> updatePayment(
            @PathVariable Long paymentId,
            @Valid @RequestBody PaymentUpdateDTO updateDTO) {

        return ResponseEntity.ok(paymentService.updatePayment(paymentId, updateDTO));
    }

    // ── DELETE (soft) ────────────────────────────────────────────────────────

    /**
     * DELETE /v1/payments/{id}
     *
     * Soft-cancels the payment (sets status = CANCELLED).
     * Hard delete is not used — the audit trail is preserved.
     * Returns 204 No Content (no body) on success.
     * Returns 422 if the payment is not PENDING.
     */
    @DeleteMapping("/{paymentId}")
    public ResponseEntity<Void> cancelPayment(
            @PathVariable Long paymentId) {

        paymentService.cancelPayment(paymentId);
        return ResponseEntity.noContent().build();
    }

    // ── AUTHORIZE ────────────────────────────────────────────────────────────

    /**
     * POST /v1/payments/{id}/authorization
     *
     * Request body:
     *   { "status": "APPROVED", "remarks": "Verified OK" }
     *   { "status": "REJECTED", "remarks": "Insufficient funds" }
     *
     * Only PENDING payments can be authorized — returns 422 otherwise.
     * Publishes payment-approved or payment-rejected event via outbox.
     */
    @PostMapping("/{paymentId}/authorization")
    public ResponseEntity<PaymentResponseDTO> authorizePayment(
            @PathVariable Long paymentId,
            @Valid @RequestBody AuthorizationRequestDTO request) {

        return ResponseEntity.ok(paymentService.authorizePayment(paymentId, request));
    }
}
