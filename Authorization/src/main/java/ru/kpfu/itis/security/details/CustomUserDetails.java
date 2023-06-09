package ru.kpfu.itis.security.details;

import org.springframework.security.core.userdetails.UserDetails;

public interface CustomUserDetails extends UserDetails {
    CustomUser getUser();
}
