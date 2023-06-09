package ru.kpfu.itis.service.impl;

import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import ru.kpfu.itis.dto.SignUpForm;
import ru.kpfu.itis.dto.UserDTO;
import ru.kpfu.itis.entity.User;
import ru.kpfu.itis.repository.UsersRepository;
import ru.kpfu.itis.service.SignUpService;

@Service
public class SignUpServiceImpl implements SignUpService {

    private final UsersRepository usersRepository;

    private final PasswordEncoder passwordEncoder;

    private final ModelMapper modelMapper;

    public SignUpServiceImpl(UsersRepository usersRepository,
                             PasswordEncoder passwordEncoder,
                             @Qualifier("customUserDetailService") UserDetailsService userDetailsService,
                             ModelMapper modelMapper) {

        this.usersRepository = usersRepository;
        this.passwordEncoder = passwordEncoder;
        this.modelMapper = modelMapper;
    }

    @Override
    public boolean userWithSuchEmailExists(String email) {
        return usersRepository.existsByEmail(email);
    }

    @Override
    public UserDTO signUp(SignUpForm signUpForm) {

        User user = User.builder()
                .name(signUpForm.getName())
                .email(signUpForm.getEmail())
                .hashedPassword(passwordEncoder.encode(signUpForm.getPassword()))
                .build();

        user = usersRepository.save(user);

        return modelMapper.map(user, UserDTO.class);
    }

}
