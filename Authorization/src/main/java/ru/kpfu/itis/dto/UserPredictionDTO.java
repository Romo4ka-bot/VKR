package ru.kpfu.itis.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserPredictionDTO implements Serializable {

    private static final long serialVersionUID = -4862926644813433707L;
    private Integer totalCholesterol;
    private LocalDateTime createdAt;
    private Long userId;
}
