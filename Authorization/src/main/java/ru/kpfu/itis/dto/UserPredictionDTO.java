package ru.kpfu.itis.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDate;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserPredictionDTO implements Serializable {

    private static final long serialVersionUID = -4862926644813433707L;

    private Float totalCholesterol;
    private LocalDate createdAt;
    private Long userId;

    public UserPredictionDTO(Float totalCholesterol) {
        this.totalCholesterol = totalCholesterol;
    }
}
