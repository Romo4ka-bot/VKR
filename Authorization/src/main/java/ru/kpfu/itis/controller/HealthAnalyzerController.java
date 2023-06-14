package ru.kpfu.itis.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import ru.kpfu.itis.dto.UserAnalyzeForm;
import ru.kpfu.itis.dto.UserPredictionDTO;
import ru.kpfu.itis.service.HealthAnalyzerService;
import ru.kpfu.itis.service.UserPredictionService;

import javax.validation.Valid;

@Controller
@RequestMapping("/analyze")
public class HealthAnalyzerController {

    private final HealthAnalyzerService healthAnalyzerService;
    private final UserPredictionService userPredictionService;

    @Autowired
    public HealthAnalyzerController(HealthAnalyzerService healthAnalyzerService, UserPredictionService userPredictionService) {
        this.healthAnalyzerService = healthAnalyzerService;
        this.userPredictionService = userPredictionService;
    }

    @PostMapping("/users/{userId}")
    public String analyzeUser(@PathVariable Long userId, @Valid UserAnalyzeForm userAnalyzeForm, BindingResult bindingResult, Model model) {
        UserPredictionDTO predict = new UserPredictionDTO();

        if (bindingResult.hasErrors()) {
            model.addAttribute("userAnalyzeForm", userAnalyzeForm);
        } else {
            predict = healthAnalyzerService.predict(userAnalyzeForm, userId);
        }

        return "redirect:/health?predict=" + Float.parseFloat(String.format("%.2f", predict.getTotalCholesterol()));
    }

    @GetMapping("/users/{userId}")
    public String getUserAnalyzeForm(Model model) {
        model.addAttribute("userAnalyzeForm", new UserAnalyzeForm());
        return "userAnalyzeForm";
    }
}
