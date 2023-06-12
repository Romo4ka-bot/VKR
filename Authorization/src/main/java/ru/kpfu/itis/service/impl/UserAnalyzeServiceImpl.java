package ru.kpfu.itis.service.impl;

import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.kpfu.itis.dto.UserAnalyzeDTO;
import ru.kpfu.itis.dto.UserAnalyzeForm;
import ru.kpfu.itis.entity.UserAnalyze;
import ru.kpfu.itis.repository.UserAnalyzeRepository;
import ru.kpfu.itis.service.UserAnalyzeService;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Optional;

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
        UserAnalyze userAnalyze = modelMapper.map(userAnalyzeForm, UserAnalyze.class);
        userAnalyze.setId(null);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        userAnalyze.setStartDate(LocalDate.parse(userAnalyzeForm.getStartDate(), formatter));
        userAnalyzeRepository.save(userAnalyze);
    }

    @Override
    public Optional<UserAnalyzeDTO> getFirstByUserId(Long userId) {
        Optional<UserAnalyze> userAnalyze = userAnalyzeRepository.findFirstByUserId(userId);
        System.out.println(userAnalyze);
        return userAnalyze.map(analyze -> modelMapper.map(analyze, UserAnalyzeDTO.class));
    }
}
