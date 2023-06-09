package ru.kpfu.itis.security.details;

import ru.kpfu.itis.entity.User;
import ru.kpfu.itis.repository.UsersRepository;
import org.modelmapper.ModelMapper;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Component;

@Component("customUserDetailService")
public class UserDetailsServiceImpl implements UserDetailsService {

    private final UsersRepository usersRepository;

    private final ModelMapper modelMapper;

    public UserDetailsServiceImpl(UsersRepository usersRepository, ModelMapper modelMapper) {
        this.usersRepository = usersRepository;
        this.modelMapper = modelMapper;
    }

    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        User user = usersRepository.getByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("User with such email not found"));

        return new UserDetailsImpl(modelMapper.map(user, CustomUser.class));
    }
}
