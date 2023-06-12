package ru.kpfu.itis.service.impl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import ru.kpfu.itis.dto.ModelCreatedDTO;
import ru.kpfu.itis.dto.UserAnalyzeDTO;
import ru.kpfu.itis.dto.UserPredictionDTO;
import ru.kpfu.itis.dto.UserAnalyzeForm;
import ru.kpfu.itis.service.HealthAnalyzerService;
import ru.kpfu.itis.service.UserAnalyzeService;
import ru.kpfu.itis.service.UserPredictionService;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Objects;
import java.util.Optional;

@Service
public class HealthAnalyzerServiceImpl implements HealthAnalyzerService {

    private final RestTemplate restTemplate;
    private final UserAnalyzeService userAnalyzeService;
    private final UserPredictionService userPredictionService;

    @Value("${predservice.url}")
    private String predictionServiceUrl;

    private static final Float NEGATIVE_TOTAL_CHOLESTEROL = -1.0f;

    @Autowired
    public HealthAnalyzerServiceImpl(RestTemplate restTemplate, UserAnalyzeService userAnalyzeService, UserPredictionService userPredictionService) {
        this.restTemplate = restTemplate;
        this.userAnalyzeService = userAnalyzeService;
        this.userPredictionService = userPredictionService;
    }

    @Override
    public UserPredictionDTO predict(UserAnalyzeForm userAnalyzeForm, Long userId) {

        userAnalyzeForm.setUserId(userId);
        Optional<UserAnalyzeDTO> userAnalyzeDTO = userAnalyzeService.getFirstByUserId(userId);
        if (userAnalyzeDTO.isPresent()) {
            userAnalyzeForm.setStartDate(userAnalyzeDTO.get().getStartDate().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));

        } else {
            userAnalyzeForm.setStartDate(LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
        }

        HttpEntity<UserAnalyzeForm> request = new HttpEntity<>(userAnalyzeForm);
        ResponseEntity<ModelCreatedDTO> isModelCreatedResponse = restTemplate
                .exchange(predictionServiceUrl + "/is-created-model", HttpMethod.GET, request, ModelCreatedDTO.class);

        System.out.println(Objects.requireNonNull(isModelCreatedResponse.getBody()));
        if (isModelCreatedResponse.getBody().getStatusModelCreated().equals("True")) {
            ResponseEntity<UserPredictionDTO> predictResponse = restTemplate
                    .exchange(predictionServiceUrl + "/predict", HttpMethod.POST, request, UserPredictionDTO.class);


            UserPredictionDTO userPredictionDTO = predictResponse.getBody();
            userPredictionDTO.setCreatedAt(LocalDate.now());

            userAnalyzeService.saveUserAnalyze(userAnalyzeForm);

            userPredictionDTO.setUserId(userId);
            userPredictionDTO.setCreatedAt(LocalDate.now());
            userPredictionService.saveUserPrediction(userPredictionDTO);

            System.out.println(3);
            return userPredictionDTO;

        } else {
            return new UserPredictionDTO(NEGATIVE_TOTAL_CHOLESTEROL);
        }
    }
}
