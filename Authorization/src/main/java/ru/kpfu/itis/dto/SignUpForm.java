package ru.kpfu.itis.dto;

import ru.kpfu.itis.validation.annotation.PasswordsMatch;
import ru.kpfu.itis.validation.annotation.UniqueEmail;
import ru.kpfu.itis.validation.annotation.ValidPassword;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@PasswordsMatch(password = "password",
                verifyPassword = "verifyPassword",
                message = "Введенные пароли не совпадают")
public class SignUpForm {

    @NotBlank(message = "Данное поле не должно быть пустым")
    @Email(message = "Неверный формат адреса электронной почты")
    @UniqueEmail(message = "Пользователь с таким адресом электронной почты уже существует")
    private String email;

    @NotBlank(message = "Данное поле не должно быть пустым")
    @Size(max = 50, message = "Максимальное количество символов - 50")
    private String name;

    @NotBlank(message = "Данное поле не должно быть пустым")
    @ValidPassword(message = "Пароль должен содержать латиницу обоих регистров, хотя бы одну цифру и состоять минимум из 8 символов")
    private String password;

    @NotBlank(message = "Данное поле не должно быть пустым")
    private String verifyPassword;

}
