package ru.kpfu.itis.controller;

import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import ru.kpfu.itis.dto.UserPredictionDTO;
import ru.kpfu.itis.security.details.CustomUserDetails;
import ru.kpfu.itis.service.UserPredictionService;

import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

@Controller
@RequestMapping("/health")
public class HealthController {

    private final UserPredictionService userPredictionService;

    public HealthController(UserPredictionService userPredictionService) {
        this.userPredictionService = userPredictionService;
    }

    @GetMapping()
    public String getMyHealthPage(Model model, @AuthenticationPrincipal CustomUserDetails userDetails) {
        System.out.println(userDetails.getUser());
        System.out.println(userDetails.getUser().getId());
        List<UserPredictionDTO> userPredictionDTOS = userPredictionService.getAllByUserId(userDetails.getUser().getId());
        System.out.println("predictions:");
        for (UserPredictionDTO userPredictionDTO : userPredictionDTOS) {
            System.out.println(userPredictionDTO);
        }
        List<Integer> listOfTotalCholesterol = new ArrayList<>();
        List<String> listOfDates = new ArrayList<>();

        for (UserPredictionDTO userPredictionDTO: userPredictionDTOS) {
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
        return "myHealth";
    }
}
