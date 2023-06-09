package ru.kpfu.itis.service;

import ru.kpfu.itis.dto.UserPredictionDTO;

import java.util.List;

public interface UserPredictionService {
    void saveUserPrediction(UserPredictionDTO userPredictionDTO);

    List<UserPredictionDTO> getAllByUserId(Long userId);
}
