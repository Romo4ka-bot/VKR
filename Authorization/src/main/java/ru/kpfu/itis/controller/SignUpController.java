package ru.kpfu.itis.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.validation.ObjectError;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import ru.kpfu.itis.dto.SignUpForm;
import ru.kpfu.itis.dto.UserDTO;
import ru.kpfu.itis.service.SignUpService;

import javax.validation.Valid;

@Controller
@RequestMapping("signUp")
public class SignUpController {

    @Autowired
    private SignUpService signUpService;

    @GetMapping
    public String getSignUpPage(Model model) {
        model.addAttribute("signUpForm", new SignUpForm());
        return "signUp";
    }

    @PostMapping
    public String signUp(@Valid SignUpForm signUpForm,
                         BindingResult bindingResult,
                         Model model) {
        if (bindingResult.hasErrors()) {

            if (bindingResult.hasGlobalErrors()) {
                ObjectError passwordMismatchError = bindingResult.getGlobalError();
                model.addAttribute("passwordMismatch", passwordMismatchError.getDefaultMessage());
            }
            model.addAttribute("signUpForm", signUpForm);
            return "signUp";
        }
        else {
            signUpService.signUp(signUpForm);
            return "redirect:/login";
        }
    }

}
