package ru.kpfu.itis.repository;

import ru.kpfu.itis.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UsersRepository extends JpaRepository<User, Long> {

    Optional<User> getByEmail(String email);

    boolean existsByEmail(String email);
}
