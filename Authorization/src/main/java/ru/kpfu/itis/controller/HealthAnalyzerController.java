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
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

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
        if (bindingResult.hasErrors()) {
            model.addAttribute("userAnalyzeForm", userAnalyzeForm);
        } else {

            System.out.println(userAnalyzeForm.toString());
            UserPredictionDTO predict = healthAnalyzerService.predict(userAnalyzeForm, userId);
            model.addAttribute("predict", predict);

            List<UserPredictionDTO> userPredictionDTOS = userPredictionService.getAllByUserId(userId);
            List<Integer> listOfTotalCholesterol = new ArrayList<>();
            List<String> listOfDates = new ArrayList<>();

            for (UserPredictionDTO userPredictionDTO : userPredictionDTOS) {
                listOfTotalCholesterol.add(userPredictionDTO.getTotalCholesterol());
                listOfDates.add(userPredictionDTO.getCreatedAt().format(DateTimeFormatter.ofPattern("dd-MM-yyyy")));
            }

            for (UserPredictionDTO userPredictionDTO : userPredictionDTOS) {
                System.out.println(userPredictionDTO);
            }
            System.out.println();
            for (Integer userPredictionDTO : listOfTotalCholesterol) {
                System.out.println(userPredictionDTO);
            }
            System.out.println();
            for (String userPredictionDTO : listOfDates) {
                System.out.println(userPredictionDTO);
            }

            model.addAttribute("listOfTotalCholesterol", listOfTotalCholesterol);
            model.addAttribute("listOfDates", listOfDates);
        }
        return "myHealth";
    }

    @GetMapping("/users/{userId}")
    public String getUserAnalyzeForm(Model model) {
        model.addAttribute("userAnalyzeForm", new UserAnalyzeForm());
        return "userAnalyzeForm";
    }
}
