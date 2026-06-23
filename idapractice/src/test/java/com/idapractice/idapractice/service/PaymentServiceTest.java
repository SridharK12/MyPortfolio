package com.idapractice.idapractice.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.idapractice.idapractice.dto.AuthorizationRequestDTO;
import com.idapractice.idapractice.dto.PaymentDTO;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.dto.PaymentUpdateDTO;
import com.idapractice.idapractice.entity.Payment;
import com.idapractice.idapractice.enums.PaymentStatus;
import com.idapractice.idapractice.exception.InvalidPaymentOperationException;
import com.idapractice.idapractice.exception.PaymentNotFoundException;
import com.idapractice.idapractice.outbox.PaymentOutboxRepository;
import com.idapractice.idapractice.repository.PaymentRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("PaymentService")
class PaymentServiceTest {

    @Mock private PaymentRepository      paymentRepository;
    @Mock private PaymentOutboxRepository outboxRepository;

    private PaymentService paymentService;

    @BeforeEach
    void setUp() {
        ObjectMapper objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule());
        paymentService = new PaymentService(paymentRepository, outboxRepository, objectMapper);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private Payment pendingPayment(Long id) {
        Payment p = new Payment();
        p.setPaymentId(id);
        p.setFromAccount("ACC001");
        p.setToAccount("ACC002");
        p.setAmount(new BigDecimal("5000.00"));
        p.setStatus(PaymentStatus.PENDING);
        p.setVersion(0L);
        return p;
    }

    private PaymentDTO validDTO() {
        PaymentDTO dto = new PaymentDTO();
        dto.setFromAccount("ACC001");
        dto.setToAccount("ACC002");
        dto.setAmount(new BigDecimal("5000.00"));
        dto.setRemarks("Test payment");
        return dto;
    }

    // ── CREATE ───────────────────────────────────────────────────────────────

    @Nested @DisplayName("createPayment")
    class CreatePayment {

        @Test
        @DisplayName("should persist with PENDING status and write to outbox — NOT directly to Kafka")
        void shouldCreatePendingPaymentAndWriteOutbox() {
            Payment saved = pendingPayment(1L);
            when(paymentRepository.save(any(Payment.class))).thenReturn(saved);

            PaymentResponseDTO response = paymentService.createPayment(validDTO(), null);

            assertThat(response.getStatus()).isEqualTo(PaymentStatus.PENDING);
            assertThat(response.getPaymentId()).isEqualTo(1L);

            // Service must write to outbox, NOT call kafkaTemplate directly
            verify(outboxRepository, times(1)).save(any());
        }

        @Test
        @DisplayName("should return existing payment when idempotency key already used")
        void shouldReturnExistingPaymentForDuplicateIdempotencyKey() {
            Payment existing = pendingPayment(42L);
            when(paymentRepository.findByIdempotencyKey("key-abc"))
                    .thenReturn(Optional.of(existing));

            PaymentResponseDTO response = paymentService.createPayment(validDTO(), "key-abc");

            assertThat(response.getPaymentId()).isEqualTo(42L);
            // Should NOT save a new payment or write to outbox
            verify(paymentRepository, never()).save(any());
            verify(outboxRepository, never()).save(any());
        }

        @Test
        @DisplayName("should create new payment when idempotency key is new")
        void shouldCreateNewPaymentForNewIdempotencyKey() {
            when(paymentRepository.findByIdempotencyKey("key-xyz"))
                    .thenReturn(Optional.empty());
            Payment saved = pendingPayment(5L);
            when(paymentRepository.save(any())).thenReturn(saved);

            paymentService.createPayment(validDTO(), "key-xyz");

            ArgumentCaptor<Payment> captor = ArgumentCaptor.forClass(Payment.class);
            verify(paymentRepository).save(captor.capture());
            assertThat(captor.getValue().getIdempotencyKey()).isEqualTo("key-xyz");
        }
    }

    // ── READ ─────────────────────────────────────────────────────────────────

    @Nested @DisplayName("getPaymentById")
    class GetById {

        @Test
        @DisplayName("should return DTO when payment exists")
        void shouldReturnPayment() {
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(pendingPayment(1L)));

            PaymentResponseDTO result = paymentService.getPaymentById(1L);
            assertThat(result.getPaymentId()).isEqualTo(1L);
        }

        @Test
        @DisplayName("should throw PaymentNotFoundException when id not found")
        void shouldThrowWhenNotFound() {
            when(paymentRepository.findById(99L)).thenReturn(Optional.empty());

            assertThatThrownBy(() -> paymentService.getPaymentById(99L))
                    .isInstanceOf(PaymentNotFoundException.class)
                    .hasMessageContaining("99");
        }
    }

    // ── UPDATE ───────────────────────────────────────────────────────────────

    @Nested @DisplayName("updatePayment")
    class UpdatePayment {

        @Test
        @DisplayName("should update fields on a PENDING payment")
        void shouldUpdatePendingPayment() {
            Payment pending = pendingPayment(1L);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(pending));
            when(paymentRepository.save(any())).thenReturn(pending);

            PaymentUpdateDTO dto = new PaymentUpdateDTO();
            dto.setAmount(new BigDecimal("9999.00"));
            dto.setRemarks("Updated");

            PaymentResponseDTO result = paymentService.updatePayment(1L, dto);

            assertThat(result).isNotNull();
            verify(outboxRepository, times(1)).save(any());
        }

        @Test
        @DisplayName("should throw InvalidPaymentOperationException when payment is APPROVED")
        void shouldThrowWhenApproved() {
            Payment approved = pendingPayment(1L);
            approved.setStatus(PaymentStatus.APPROVED);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(approved));

            assertThatThrownBy(() -> paymentService.updatePayment(1L, new PaymentUpdateDTO()))
                    .isInstanceOf(InvalidPaymentOperationException.class)
                    .hasMessageContaining("APPROVED");
        }

        @Test
        @DisplayName("should throw InvalidPaymentOperationException when payment is REJECTED")
        void shouldThrowWhenRejected() {
            Payment rejected = pendingPayment(1L);
            rejected.setStatus(PaymentStatus.REJECTED);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(rejected));

            assertThatThrownBy(() -> paymentService.updatePayment(1L, new PaymentUpdateDTO()))
                    .isInstanceOf(InvalidPaymentOperationException.class)
                    .hasMessageContaining("REJECTED");
        }
    }

    // ── AUTHORIZE ─────────────────────────────────────────────────────────────

    @Nested @DisplayName("authorizePayment")
    class AuthorizePayment {

        @Test
        @DisplayName("should approve a PENDING payment and write to payment-approved outbox topic")
        void shouldApprovePayment() {
            Payment pending = pendingPayment(1L);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(pending));
            when(paymentRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            AuthorizationRequestDTO req = new AuthorizationRequestDTO();
            req.setStatus("APPROVED");
            req.setRemarks("Looks good");

            paymentService.authorizePayment(1L, req);

            ArgumentCaptor<Payment> paymentCaptor = ArgumentCaptor.forClass(Payment.class);
            verify(paymentRepository).save(paymentCaptor.capture());
            assertThat(paymentCaptor.getValue().getStatus()).isEqualTo(PaymentStatus.APPROVED);

            ArgumentCaptor<com.idapractice.idapractice.entity.PaymentOutboxEvent> outboxCaptor =
                    ArgumentCaptor.forClass(com.idapractice.idapractice.entity.PaymentOutboxEvent.class);
            verify(outboxRepository).save(outboxCaptor.capture());
            assertThat(outboxCaptor.getValue().getTopic()).isEqualTo(PaymentService.TOPIC_APPROVED);
        }

        @Test
        @DisplayName("should reject a PENDING payment and write to payment-rejected outbox topic")
        void shouldRejectPayment() {
            Payment pending = pendingPayment(2L);
            when(paymentRepository.findById(2L)).thenReturn(Optional.of(pending));
            when(paymentRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            AuthorizationRequestDTO req = new AuthorizationRequestDTO();
            req.setStatus("REJECTED");
            req.setRemarks("Insufficient funds");

            paymentService.authorizePayment(2L, req);

            ArgumentCaptor<com.idapractice.idapractice.entity.PaymentOutboxEvent> captor =
                    ArgumentCaptor.forClass(com.idapractice.idapractice.entity.PaymentOutboxEvent.class);
            verify(outboxRepository).save(captor.capture());
            assertThat(captor.getValue().getTopic()).isEqualTo(PaymentService.TOPIC_REJECTED);
        }

        @Test
        @DisplayName("should throw when payment is already CANCELLED")
        void shouldThrowWhenCancelled() {
            Payment cancelled = pendingPayment(1L);
            cancelled.setStatus(PaymentStatus.CANCELLED);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(cancelled));

            AuthorizationRequestDTO req = new AuthorizationRequestDTO();
            req.setStatus("APPROVED");

            assertThatThrownBy(() -> paymentService.authorizePayment(1L, req))
                    .isInstanceOf(InvalidPaymentOperationException.class);
        }
    }

    // ── DELETE ────────────────────────────────────────────────────────────────

    @Nested @DisplayName("cancelPayment")
    class CancelPayment {

        @Test
        @DisplayName("should soft-delete a PENDING payment by setting status to CANCELLED")
        void shouldCancelPendingPayment() {
            Payment pending = pendingPayment(1L);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(pending));
            when(paymentRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            paymentService.cancelPayment(1L);

            ArgumentCaptor<Payment> captor = ArgumentCaptor.forClass(Payment.class);
            verify(paymentRepository).save(captor.capture());
            assertThat(captor.getValue().getStatus()).isEqualTo(PaymentStatus.CANCELLED);
            verify(outboxRepository, times(1)).save(any());
        }

        @Test
        @DisplayName("should throw InvalidPaymentOperationException when cancelling an APPROVED payment")
        void shouldThrowWhenCancellingApproved() {
            Payment approved = pendingPayment(1L);
            approved.setStatus(PaymentStatus.APPROVED);
            when(paymentRepository.findById(1L)).thenReturn(Optional.of(approved));

            assertThatThrownBy(() -> paymentService.cancelPayment(1L))
                    .isInstanceOf(InvalidPaymentOperationException.class)
                    .hasMessageContaining("APPROVED");
        }
    }
}
