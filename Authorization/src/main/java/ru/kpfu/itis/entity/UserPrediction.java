package ru.kpfu.itis.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import javax.persistence.*;
import java.time.LocalDate;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Entity
@Table(name = "user_prediction")
public class UserPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Float totalCholesterol;

    @CreationTimestamp
    private LocalDate createdAt;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;
}
