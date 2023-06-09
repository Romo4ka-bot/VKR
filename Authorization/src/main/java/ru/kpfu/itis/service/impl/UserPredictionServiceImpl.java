package ru.kpfu.itis.service.impl;

import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.kpfu.itis.dto.UserPredictionDTO;
import ru.kpfu.itis.entity.UserPrediction;
import ru.kpfu.itis.repository.UserPredictionRepository;
import ru.kpfu.itis.service.UserPredictionService;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class UserPredictionServiceImpl implements UserPredictionService {

    private final UserPredictionRepository userPredictionRepository;
    private final ModelMapper modelMapper;

    @Autowired
    public UserPredictionServiceImpl(UserPredictionRepository userPredictionRepository, ModelMapper modelMapper) {
        this.userPredictionRepository = userPredictionRepository;
        this.modelMapper = modelMapper;
    }

    @Override
    public void saveUserPrediction(UserPredictionDTO userPredictionDTO) {
        userPredictionRepository.save(modelMapper.map(userPredictionDTO, UserPrediction.class));
    }

    @Override
    public List<UserPredictionDTO> getAllByUserId(Long userId) {
        return userPredictionRepository.findFirst5ByUserId(userId).stream().map(prediction -> modelMapper.map(prediction, UserPredictionDTO.class)).collect(Collectors.toList());
    }
}
