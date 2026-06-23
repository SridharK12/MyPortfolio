package com.idapractice.idapractice.exception;

public class AccountNotFoundException extends RuntimeException{
		
	public AccountNotFoundException(String accountNumber)
	{
		super ("Account Number "+ accountNumber + " Does not exist in DB");
	}

}
