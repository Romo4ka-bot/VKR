package ru.kpfu.itis.security.details;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * @author Anvar Khasanov
 * student of ITIS KFU
 * group 11-905
 */

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CustomUser implements Serializable {
    private static final long serialVersionUID = 4437060923718907313L;

    private Long id;
    private String name;
    private String email;
    private String hashedPassword;
    private String avatar;
    private String confirmCode;
}
