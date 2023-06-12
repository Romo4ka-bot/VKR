package ru.kpfu.itis.controller;

import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import ru.kpfu.itis.dto.UserPredictionDTO;
import ru.kpfu.itis.security.details.CustomUserDetails;
import ru.kpfu.itis.service.UserPredictionService;

import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@Controller
@RequestMapping("/health")
public class HealthController {

    private final UserPredictionService userPredictionService;

    private static final Float NEGATIVE_TOTAL_CHOLESTEROL = -1.0f;
    private static final Float ZERO_TOTAL_CHOLESTEROL = 0.0f;

    public HealthController(UserPredictionService userPredictionService) {
        this.userPredictionService = userPredictionService;
    }

    @GetMapping()
    public String getMyHealthPage(Model model, @AuthenticationPrincipal CustomUserDetails userDetails, @RequestParam(value = "predict") Float predict) {

        List<UserPredictionDTO> userPredictionDTOS = userPredictionService.getAllByUserId(userDetails.getUser().getId());

        List<Float> listOfTotalCholesterol = new ArrayList<>();
        List<String> listOfDates = new ArrayList<>();

        for (UserPredictionDTO userPredictionDTO : userPredictionDTOS) {
            listOfTotalCholesterol.add(Float.parseFloat(String.format("%.2f", userPredictionDTO.getTotalCholesterol())));
            listOfDates.add(userPredictionDTO.getCreatedAt().format(DateTimeFormatter.ofPattern("dd-MM-yyyy")));
        }

        if (Objects.equals(predict, NEGATIVE_TOTAL_CHOLESTEROL)) {
            model.addAttribute("modelNotCreated", "Модель еще не готова для предсказания повторите попытку чуть позже");
        } else {
            if (!Objects.equals(predict, ZERO_TOTAL_CHOLESTEROL)) {
                model.addAttribute("predict", Float.parseFloat(String.format("%.2f", predict)));
            }
            if (listOfTotalCholesterol.size() == 0) {
                model.addAttribute("noPredictions", "Видимо вы еще не делали проверки своего уровня холестерина, " +
                        "скорее сделайте и сможете следить в реальном времени за его изменением");
            }
        }

        model.addAttribute("listOfTotalCholesterol", listOfTotalCholesterol);
        model.addAttribute("listOfDates", listOfDates);
        return "myHealth";
    }
}
