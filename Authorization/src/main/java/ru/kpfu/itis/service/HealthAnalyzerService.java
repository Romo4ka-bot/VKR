package ru.kpfu.itis.service;

import ru.kpfu.itis.dto.UserAnalyzeForm;
import ru.kpfu.itis.dto.UserPredictionDTO;

public interface HealthAnalyzerService {
    UserPredictionDTO predict(UserAnalyzeForm userAnalyzeForm, Long userId);
}
