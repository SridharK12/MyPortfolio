package com.example.customerapi.entity;

import jakarta.persistence.*;

@Entity
@Table(name = "customer_additional_details")
public class CustomerAdditionalDetails {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "customer_id", nullable = false, unique = true)
    private Customer customer;

    @Column(name = "addl_dtl1")
    private String addlDtl1;

    @Column(name = "addl_dtl2")
    private String addlDtl2;

    @Column(name = "addl_dtl3")
    private String addlDtl3;

    protected CustomerAdditionalDetails() {}

    public Long getId() { return id; }
    public Customer getCustomer() { return customer; }
    public void setCustomer(Customer customer) { this.customer = customer; }
    public String getAddlDtl1() { return addlDtl1; }
    public void setAddlDtl1(String addlDtl1) { this.addlDtl1 = addlDtl1; }
    public String getAddlDtl2() { return addlDtl2; }
    public void setAddlDtl2(String addlDtl2) { this.addlDtl2 = addlDtl2; }
    public String getAddlDtl3() { return addlDtl3; }
    public void setAddlDtl3(String addlDtl3) { this.addlDtl3 = addlDtl3; }
}
