package ru.kpfu.itis.entity;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.*;

import javax.persistence.*;
import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Entity
@Table(name = "account")
@ToString(exclude = {"userAnalyzes", "userPredictions"})
@EqualsAndHashCode(exclude = {"userAnalyzes", "userPredictions"})
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 50)
    private String name;

    @Column(length = 30, unique = true)
    private String email;

    private String hashedPassword;

    @JsonIgnore
    @OneToMany(mappedBy = "user", fetch = FetchType.EAGER)
    private Set<UserAnalyze> userAnalyzes;

    @JsonIgnore
    @OneToMany(mappedBy = "user", fetch = FetchType.EAGER)
    private Set<UserPrediction> userPredictions;
}
