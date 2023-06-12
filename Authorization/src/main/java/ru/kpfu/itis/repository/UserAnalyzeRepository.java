package ru.kpfu.itis.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.kpfu.itis.entity.UserAnalyze;

import java.util.Optional;

public interface UserAnalyzeRepository extends JpaRepository<UserAnalyze, Long> {
    Optional<UserAnalyze> findFirstByUserId(Long userId);
}
