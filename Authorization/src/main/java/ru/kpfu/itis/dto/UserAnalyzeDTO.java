package ru.kpfu.itis.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserAnalyzeDTO {

    private String primarySecondaryCvrm;
    private String hypertension;
    private String gender;
    private String smokingStatus;
    private String organisationName;
    private Integer glucoseFasting;
    private Integer systolicBloodPressure;
    private Integer diastolicBloodPressure;
    private Integer bmi;
    private Integer age;
    private Long userId;
    private LocalDate startDate;
}
