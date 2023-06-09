package ru.kpfu.itis.validation.validator;

import ru.kpfu.itis.service.SignUpService;
import ru.kpfu.itis.validation.annotation.UniqueEmail;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;

@Component
public class UniqueEmailValidator implements ConstraintValidator<UniqueEmail, String> {

    @Autowired
    private SignUpService signUpService;

    @Override
    public boolean isValid(String email, ConstraintValidatorContext constraintValidatorContext) {
        return !signUpService.userWithSuchEmailExists(email);
    }

}
