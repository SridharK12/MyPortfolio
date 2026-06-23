package com.idapractice.idapractice.service;
import java.math.BigDecimal;	

import com.idapractice.idapractice.repository.PaymentRepository;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import com.idapractice.idapractice.dto.AuthorizationRequestDTO;
import com.idapractice.idapractice.dto.PaymentDTO;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.entity.Payment;

@Service
public class PaymentService {

	// Payment lifecycle statuses
	private static final String STATUS_PENDING  = "PENDING";
	private static final String STATUS_APPROVED = "APPROVED";
	private static final String STATUS_REJECTED = "REJECTED";

	// Kafka topics
	private static final String TOPIC_PAYMENT_APPROVED = "payment-approved";
	private static final String TOPIC_PAYMENT_REJECTED = "payment-rejected";

	private final PaymentRepository paymentRepository;
	private final KafkaTemplate<String, Object> kafkaTemplate;

	public PaymentService(PaymentRepository paymentRepository,
	                      KafkaTemplate<String, Object> kafkaTemplate)
	{
		this.paymentRepository = paymentRepository;
		this.kafkaTemplate = kafkaTemplate;
	}

	public PaymentResponseDTO savePaymentAfterValidations(PaymentDTO paymentDTO) {

	    // Business validations
	    if (paymentDTO.getAmount() == null ||
	        paymentDTO.getAmount().compareTo(BigDecimal.ZERO) <= 0) {
	        throw new IllegalArgumentException("Amount must be greater than zero");
	    }

	    if (paymentDTO.getFromAccount() == null ||
	        paymentDTO.getFromAccount().isBlank()) {
	        throw new IllegalArgumentException("From account is mandatory");
	    }

	    if (paymentDTO.getToAccount() == null ||
	        paymentDTO.getToAccount().isBlank()) {
	        throw new IllegalArgumentException("To account is mandatory");
	    }

	    // DTO -> Entity
	    Payment payment = new Payment();
	    payment.setFromAccount(paymentDTO.getFromAccount());
	    payment.setToAccount(paymentDTO.getToAccount());
	    payment.setAmount(paymentDTO.getAmount());
	    payment.setRemarks(paymentDTO.getRemarks());

	    // A new payment always starts its lifecycle as PENDING,
	    // regardless of any status supplied by the caller.
	    payment.setStatus(STATUS_PENDING);

	    // Save
	    Payment savedPayment = paymentRepository.save(payment);

	    // Entity -> Response DTO
	    return buildResponse(savedPayment);
	}

    public PaymentResponseDTO authorizePayment(
            Long paymentId,
            AuthorizationRequestDTO request) {

        Payment payment = paymentRepository.findById(paymentId)
                .orElseThrow(() ->
                        new RuntimeException("Payment not found"));

        // Normalise the incoming decision so "approved"/"Approved"/"APPROVED"
        // are all treated the same.
        String decision = request.getStatus() == null
                ? null
                : request.getStatus().trim().toUpperCase();

        if (!STATUS_APPROVED.equals(decision) &&
            !STATUS_REJECTED.equals(decision)) {

            throw new IllegalArgumentException(
                    "Status must be APPROVED or REJECTED");
        }

        payment.setStatus(decision);

        if (request.getRemarks() != null) {
            payment.setRemarks(request.getRemarks());
        }

        Payment savedPayment = paymentRepository.save(payment);

        PaymentResponseDTO response = buildResponse(savedPayment);

        // Publish the payment record to the topic matching the decision.
        // Keyed by paymentId so all events for one payment land on the
        // same partition (preserves ordering per payment).
        String key = String.valueOf(savedPayment.getPaymentId());

        if (STATUS_APPROVED.equals(decision)) {
            kafkaTemplate.send(TOPIC_PAYMENT_APPROVED, key, response);
        } else {
            kafkaTemplate.send(TOPIC_PAYMENT_REJECTED, key, response);
        }

        return response;
    }

    private PaymentResponseDTO buildResponse(Payment payment) {

        PaymentResponseDTO response =
                new PaymentResponseDTO();

        response.setPaymentId(payment.getPaymentId());
        response.setFromAccount(payment.getFromAccount());
        response.setToAccount(payment.getToAccount());
        response.setAmount(payment.getAmount());
        response.setRemarks(payment.getRemarks());
        response.setStatus(payment.getStatus());

        return response;
    }

}
