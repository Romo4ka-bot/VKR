package ru.kpfu.itis.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Entity
@Table(name = "user_analyze")
public class UserAnalyze {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 10)
    private String primarySecondaryCvrm;

    @Column(length = 10)
    private String hypertension;

    @Column(length = 10)
    private String gender;

    @Column(length = 10)
    private String smokingStatus;

    @Column(length = 50)
    private String organisationName;

    private Integer glucoseFasting;
    private Integer systolicBloodPressure;
    private Integer diastolicBloodPressure;
    private Integer bmi;
    private Integer age;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;
}
