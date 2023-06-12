package ru.kpfu.itis.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class StaticPagesController {

    @GetMapping("/login_or_sign_up")
    public String getLoginOrSignUpPage() {
        return "login";
    }

    @GetMapping("/about")
    public String getAboutPage() {
        return "about";
    }
}
