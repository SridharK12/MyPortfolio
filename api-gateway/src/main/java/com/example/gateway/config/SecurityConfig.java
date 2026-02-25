package com.example.gateway.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;

@Configuration
@EnableWebFluxSecurity
public class SecurityConfig {

    @Bean
    public SecurityWebFilterChain securityWebFilterChain(ServerHttpSecurity http) {

        return http
            .csrf(ServerHttpSecurity.CsrfSpec::disable)

            .authorizeExchange(exchanges -> exchanges
                // ✅ Allow actuator endpoints (observability)
                .pathMatchers("/actuator/**").permitAll()

                // ✅ Secure business APIs
                .pathMatchers("/api/**").authenticated()
                .pathMatchers("/status/**").authenticated()

                // ❌ Block everything else
                .anyExchange().denyAll()
            )

            // ✅ Basic auth (from application.yml)
            .httpBasic(Customizer.withDefaults())

            // ❌ Disable form login (not needed for APIs)
            .formLogin(ServerHttpSecurity.FormLoginSpec::disable)

            .build();
    }
}