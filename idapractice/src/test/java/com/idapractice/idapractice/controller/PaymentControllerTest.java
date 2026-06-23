package com.idapractice.idapractice.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.idapractice.idapractice.dto.PaymentDTO;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.dto.PaymentUpdateDTO;
import com.idapractice.idapractice.dto.AuthorizationRequestDTO;
import com.idapractice.idapractice.enums.PaymentStatus;
import com.idapractice.idapractice.exception.InvalidPaymentOperationException;
import com.idapractice.idapractice.exception.PaymentNotFoundException;
import com.idapractice.idapractice.service.PaymentService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.math.BigDecimal;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(PaymentController.class)
@DisplayName("PaymentController")
class PaymentControllerTest {

    @Autowired private MockMvc mockMvc;
    @MockitoBean private PaymentService paymentService;

    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private PaymentResponseDTO sampleResponse(Long id) {
        PaymentResponseDTO dto = new PaymentResponseDTO();
        dto.setPaymentId(id);
        dto.setFromAccount("ACC001");
        dto.setToAccount("ACC002");
        dto.setAmount(new BigDecimal("5000.00"));
        dto.setStatus(PaymentStatus.PENDING);
        dto.setVersion(0L);
        return dto;
    }

    private String json(Object obj) throws Exception {
        return objectMapper.writeValueAsString(obj);
    }

    // ── POST /v1/payments ─────────────────────────────────────────────────────

    @Nested @DisplayName("POST /v1/payments")
    class CreatePayment {

        @Test
        @DisplayName("should return 201 Created with valid body")
        void shouldReturn201() throws Exception {
            PaymentDTO dto = new PaymentDTO();
            dto.setFromAccount("ACC001");
            dto.setToAccount("ACC002");
            dto.setAmount(new BigDecimal("5000.00"));

            when(paymentService.createPayment(any(), isNull()))
                    .thenReturn(sampleResponse(1L));

            mockMvc.perform(post("/v1/payments")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(dto)))
                    .andExpect(status().isCreated())
                    .andExpect(jsonPath("$.paymentId").value(1))
                    .andExpect(jsonPath("$.status").value("PENDING"));
        }

        @Test
        @DisplayName("should return 400 when fromAccount is blank")
        void shouldReturn400WhenFromAccountMissing() throws Exception {
            PaymentDTO dto = new PaymentDTO();
            dto.setToAccount("ACC002");
            dto.setAmount(new BigDecimal("5000.00"));
            // fromAccount deliberately omitted

            mockMvc.perform(post("/v1/payments")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(dto)))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error").value("Validation Failed"))
                    .andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("fromAccount")));
        }

        @Test
        @DisplayName("should return 400 when amount is null")
        void shouldReturn400WhenAmountNull() throws Exception {
            PaymentDTO dto = new PaymentDTO();
            dto.setFromAccount("ACC001");
            dto.setToAccount("ACC002");
            // amount deliberately omitted

            mockMvc.perform(post("/v1/payments")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(dto)))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error").value("Validation Failed"));
        }

        @Test
        @DisplayName("should return 400 when body is missing")
        void shouldReturn400WhenBodyMissing() throws Exception {
            mockMvc.perform(post("/v1/payments")
                            .contentType(MediaType.APPLICATION_JSON))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error").value("Bad Request"));
        }

        @Test
        @DisplayName("should honour X-Idempotency-Key header")
        void shouldPassIdempotencyKeyToService() throws Exception {
            PaymentDTO dto = new PaymentDTO();
            dto.setFromAccount("ACC001");
            dto.setToAccount("ACC002");
            dto.setAmount(new BigDecimal("5000.00"));

            when(paymentService.createPayment(any(), eq("my-key-123")))
                    .thenReturn(sampleResponse(7L));

            mockMvc.perform(post("/v1/payments")
                            .header("X-Idempotency-Key", "my-key-123")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(dto)))
                    .andExpect(status().isCreated())
                    .andExpect(jsonPath("$.paymentId").value(7));
        }
    }

    // ── GET /v1/payments ──────────────────────────────────────────────────────

    @Nested @DisplayName("GET /v1/payments")
    class GetAllPayments {

        @Test
        @DisplayName("should return 200 with paginated results")
        void shouldReturnPage() throws Exception {
            Page<PaymentResponseDTO> page = new PageImpl<>(List.of(sampleResponse(1L)));
            when(paymentService.getAllPayments(isNull(), isNull(), eq(0), eq(20)))
                    .thenReturn(page);

            mockMvc.perform(get("/v1/payments"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.content[0].paymentId").value(1))
                    .andExpect(jsonPath("$.totalElements").value(1));
        }

        @Test
        @DisplayName("should pass status and fromAccount filters to service")
        void shouldPassFilters() throws Exception {
            when(paymentService.getAllPayments(eq("PENDING"), eq("ACC001"), eq(0), eq(10)))
                    .thenReturn(Page.empty());

            mockMvc.perform(get("/v1/payments")
                            .param("status", "PENDING")
                            .param("fromAccount", "ACC001")
                            .param("size", "10"))
                    .andExpect(status().isOk());
        }
    }

    // ── GET /v1/payments/{id} ─────────────────────────────────────────────────

    @Nested @DisplayName("GET /v1/payments/{id}")
    class GetById {

        @Test
        @DisplayName("should return 200 with payment body")
        void shouldReturn200() throws Exception {
            when(paymentService.getPaymentById(1L)).thenReturn(sampleResponse(1L));

            mockMvc.perform(get("/v1/payments/1"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.paymentId").value(1));
        }

        @Test
        @DisplayName("should return 404 with ErrorResponseDTO when payment not found")
        void shouldReturn404() throws Exception {
            when(paymentService.getPaymentById(99L))
                    .thenThrow(new PaymentNotFoundException(99L));

            mockMvc.perform(get("/v1/payments/99"))
                    .andExpect(status().isNotFound())
                    .andExpect(jsonPath("$.status").value(404))
                    .andExpect(jsonPath("$.error").value("Not Found"))
                    .andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("99")))
                    .andExpect(jsonPath("$.path").value("/v1/payments/99"))
                    .andExpect(jsonPath("$.timestamp").exists());
        }

        @Test
        @DisplayName("should return 400 when id is not a number")
        void shouldReturn400ForBadPathVar() throws Exception {
            mockMvc.perform(get("/v1/payments/not-a-number"))
                    .andExpect(status().isBadRequest());
        }
    }

    // ── PUT /v1/payments/{id} ─────────────────────────────────────────────────

    @Nested @DisplayName("PUT /v1/payments/{id}")
    class UpdatePayment {

        @Test
        @DisplayName("should return 200 on successful update")
        void shouldReturn200() throws Exception {
            PaymentUpdateDTO dto = new PaymentUpdateDTO();
            dto.setAmount(new BigDecimal("9999.00"));

            when(paymentService.updatePayment(eq(1L), any(PaymentUpdateDTO.class)))
                    .thenReturn(sampleResponse(1L));

            mockMvc.perform(put("/v1/payments/1")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(dto)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.paymentId").value(1));
        }

        @Test
        @DisplayName("should return 422 when payment is not PENDING")
        void shouldReturn422WhenNotPending() throws Exception {
            when(paymentService.updatePayment(eq(1L), any()))
                    .thenThrow(new InvalidPaymentOperationException("Cannot update payment 1 — current status is 'APPROVED'"));

            mockMvc.perform(put("/v1/payments/1")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(new PaymentUpdateDTO())))
                    .andExpect(status().isUnprocessableEntity())
                    .andExpect(jsonPath("$.status").value(422))
                    .andExpect(jsonPath("$.error").value("Unprocessable Entity"));
        }
    }

    // ── DELETE /v1/payments/{id} ──────────────────────────────────────────────

    @Nested @DisplayName("DELETE /v1/payments/{id}")
    class DeletePayment {

        @Test
        @DisplayName("should return 204 No Content on successful cancel")
        void shouldReturn204() throws Exception {
            doNothing().when(paymentService).cancelPayment(1L);

            mockMvc.perform(delete("/v1/payments/1"))
                    .andExpect(status().isNoContent());
        }

        @Test
        @DisplayName("should return 404 when payment not found")
        void shouldReturn404WhenNotFound() throws Exception {
            doThrow(new PaymentNotFoundException(55L)).when(paymentService).cancelPayment(55L);

            mockMvc.perform(delete("/v1/payments/55"))
                    .andExpect(status().isNotFound())
                    .andExpect(jsonPath("$.status").value(404));
        }

        @Test
        @DisplayName("should return 422 when payment is APPROVED")
        void shouldReturn422WhenApproved() throws Exception {
            doThrow(new InvalidPaymentOperationException("Cannot cancel payment — status is APPROVED"))
                    .when(paymentService).cancelPayment(1L);

            mockMvc.perform(delete("/v1/payments/1"))
                    .andExpect(status().isUnprocessableEntity());
        }
    }

    // ── POST /v1/payments/{id}/authorization ──────────────────────────────────

    @Nested @DisplayName("POST /v1/payments/{id}/authorization")
    class AuthorizePayment {

        @Test
        @DisplayName("should return 200 on APPROVED decision")
        void shouldReturn200ForApproval() throws Exception {
            AuthorizationRequestDTO req = new AuthorizationRequestDTO();
            req.setStatus("APPROVED");
            req.setRemarks("All clear");

            PaymentResponseDTO approved = sampleResponse(1L);
            approved.setStatus(PaymentStatus.APPROVED);
            when(paymentService.authorizePayment(eq(1L), any())).thenReturn(approved);

            mockMvc.perform(post("/v1/payments/1/authorization")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(req)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.status").value("APPROVED"));
        }

        @Test
        @DisplayName("should return 400 when status is missing in authorization request")
        void shouldReturn400WhenStatusMissing() throws Exception {
            AuthorizationRequestDTO req = new AuthorizationRequestDTO();
            // status deliberately omitted

            mockMvc.perform(post("/v1/payments/1/authorization")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(json(req)))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.error").value("Validation Failed"));
        }
    }
}
