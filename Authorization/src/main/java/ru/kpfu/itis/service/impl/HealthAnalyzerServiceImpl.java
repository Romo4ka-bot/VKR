package ru.kpfu.itis.service.impl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import ru.kpfu.itis.dto.UserPredictionDTO;
import ru.kpfu.itis.dto.UserAnalyzeForm;
import ru.kpfu.itis.service.HealthAnalyzerService;
import ru.kpfu.itis.service.UserAnalyzeService;
import ru.kpfu.itis.service.UserPredictionService;

import java.time.LocalDateTime;

@Service
public class HealthAnalyzerServiceImpl implements HealthAnalyzerService {

    private final RestTemplate restTemplate;
    private final UserAnalyzeService userAnalyzeService;
    private final UserPredictionService userPredictionService;

    @Value("${predservice.url}")
    private String predictionServiceUrl;

    @Autowired
    public HealthAnalyzerServiceImpl(RestTemplate restTemplate, UserAnalyzeService userAnalyzeService, UserPredictionService userPredictionService) {
        this.restTemplate = restTemplate;
        this.userAnalyzeService = userAnalyzeService;
        this.userPredictionService = userPredictionService;
    }

    @Override
    public UserPredictionDTO predict(UserAnalyzeForm userAnalyzeForm, Long userId) {

        userAnalyzeForm.setUserId(userId);

        HttpEntity<UserAnalyzeForm> request = new HttpEntity<>(userAnalyzeForm);
        ResponseEntity<UserPredictionDTO> response = restTemplate
                .exchange(predictionServiceUrl + "/predict", HttpMethod.POST, request, UserPredictionDTO.class);

        UserPredictionDTO userPredictionDTO = response.getBody();

        userAnalyzeService.saveUserAnalyze(userAnalyzeForm);

        userPredictionDTO.setUserId(userId);
        userPredictionDTO.setCreatedAt(LocalDateTime.now());
        userPredictionService.saveUserPrediction(userPredictionDTO);

        return userPredictionDTO;
    }
}
