package com.idapractice.idapractice.controller;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.idapractice.idapractice.service.PaymentService;
import org.springframework.http.ResponseEntity;
import com.idapractice.idapractice.dto.PaymentDTO;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.dto.AuthorizationRequestDTO;
import java.util.List;

@RestController
@RequestMapping("/v1/payments")
public class PaymentController {

	private final PaymentService paymentService;
	
	public PaymentController(PaymentService paymentService)
	{
		this.paymentService=paymentService;	
	}
	
	@PostMapping
	
	public ResponseEntity <PaymentResponseDTO> savePayment(@RequestBody PaymentDTO payment )
	{
		return ResponseEntity.ok(paymentService.savePaymentAfterValidations(payment));
	}
	
	   @PostMapping("/{paymentId}/authorization")
	    public ResponseEntity<PaymentResponseDTO> authorizePayment(
	            @PathVariable Long paymentId,
	            @RequestBody AuthorizationRequestDTO request) {

	        return ResponseEntity.ok(
	                paymentService.authorizePayment(paymentId, request));
	    }}
