package ru.kpfu.itis.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.kpfu.itis.entity.UserPrediction;

import java.util.List;

public interface UserPredictionRepository extends JpaRepository<UserPrediction, Long> {
    List<UserPrediction> findFirst5ByUserId(Long userId);
}
