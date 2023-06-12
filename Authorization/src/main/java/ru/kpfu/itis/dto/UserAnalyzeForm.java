package ru.kpfu.itis.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.validation.constraints.Max;
import javax.validation.constraints.Min;
import javax.validation.constraints.NotBlank;
import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserAnalyzeForm {

    @NotBlank(message = "Данное поле не должно быть пустым")
    private String primarySecondaryCvrm;

    @NotBlank(message = "Данное поле не должно быть пустым")
    private String hypertension;

    @NotBlank(message = "Данное поле не должно быть пустым")
    private String gender;

    @NotBlank(message = "Данное поле не должно быть пустым")
    private String smokingStatus;

    @NotBlank(message = "Данное поле не должно быть пустым")
    private String organisationName;

    @Max(value = 25, message = "Значение должно быть меньше 26")
    @Min(value = 3, message = "Значение должно быть больше 2")
    private Integer glucoseFasting;

    @Max(value = 230, message = "Значение должно быть меньше 231")
    @Min(value = 90, message = "Значение должно быть больше 89")
    private Integer systolicBloodPressure;

    @Max(value = 120, message = "Значение должно быть меньше 121")
    @Min(value = 40, message = "Значение должно быть больше 39")
    private Integer diastolicBloodPressure;

    @Max(value = 45, message = "Значение должно быть меньше 46")
    @Min(value = 15, message = "Значение должно быть больше 14")
    private Integer bmi;

    @Max(value = 120, message = "Значение должно быть меньше 121")
    @Min(value = 1, message = "Значение должно быть больше 0")
    private Integer age;

    private Long userId;

    private String startDate;
}
