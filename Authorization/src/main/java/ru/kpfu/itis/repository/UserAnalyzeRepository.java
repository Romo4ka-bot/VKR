package ru.kpfu.itis.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.kpfu.itis.entity.UserAnalyze;

public interface UserAnalyzeRepository extends JpaRepository<UserAnalyze, Long> {
}
