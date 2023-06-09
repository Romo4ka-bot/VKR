package ru.kpfu.itis.service.impl;

import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.kpfu.itis.dto.UserAnalyzeForm;
import ru.kpfu.itis.entity.UserAnalyze;
import ru.kpfu.itis.repository.UserAnalyzeRepository;
import ru.kpfu.itis.service.UserAnalyzeService;

@Service
public class UserAnalyzeServiceImpl implements UserAnalyzeService {

    private final UserAnalyzeRepository userAnalyzeRepository;
    private final ModelMapper modelMapper;

    @Autowired
    public UserAnalyzeServiceImpl(UserAnalyzeRepository userAnalyzeRepository, ModelMapper modelMapper) {
        this.userAnalyzeRepository = userAnalyzeRepository;
        this.modelMapper = modelMapper;
    }

    @Override
    public void saveUserAnalyze(UserAnalyzeForm userAnalyzeForm) {
        userAnalyzeRepository.save(modelMapper.map(userAnalyzeForm, UserAnalyze.class));
    }
}
