package com.idapractice.idapractice.component;
import org.springframework.stereotype.Component;
import org.springframework.kafka.annotation.KafkaListener;

@Component
public class PaymentEventConsumer {

    @KafkaListener(
        topics = "payment-approved",
        groupId = "payment-group"
    )
    public void consume(String message) {
        System.out.println("Received message: " + message);
    }
}