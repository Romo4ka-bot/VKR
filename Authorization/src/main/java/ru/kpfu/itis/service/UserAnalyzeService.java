package ru.kpfu.itis.service;

import ru.kpfu.itis.dto.UserAnalyzeDTO;
import ru.kpfu.itis.dto.UserAnalyzeForm;

import java.util.List;
import java.util.Optional;

public interface UserAnalyzeService {
    void saveUserAnalyze(UserAnalyzeForm userAnalyzeForm);

    Optional<UserAnalyzeDTO> getFirstByUserId(Long userId);
}
