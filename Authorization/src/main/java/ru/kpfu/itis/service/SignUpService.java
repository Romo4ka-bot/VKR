package ru.kpfu.itis.service;

import ru.kpfu.itis.dto.SignUpForm;
import ru.kpfu.itis.dto.UserDTO;

public interface SignUpService {

    boolean userWithSuchEmailExists(String email);
    UserDTO signUp(SignUpForm signUpForm);
}
