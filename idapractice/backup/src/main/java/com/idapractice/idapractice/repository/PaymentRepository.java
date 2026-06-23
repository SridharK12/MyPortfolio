package com.idapractice.idapractice.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.idapractice.idapractice.entity.Payment;


import java.util.List;

@Repository
public interface PaymentRepository extends JpaRepository<Payment, Long> {
	
	

}
