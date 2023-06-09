package ru.kpfu.itis.interceptor;

import ru.kpfu.itis.dto.UserDTO;
import ru.kpfu.itis.security.details.CustomUserDetails;
import org.modelmapper.ModelMapper;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * @author Anvar Khasanov
 * student of ITIS KFU
 * group 11-905
 */

@Component
public class PastUserInTemplateInterceptor implements HandlerInterceptor {

    private final ModelMapper modelMapper;

    public PastUserInTemplateInterceptor(ModelMapper modelMapper) {
        this.modelMapper = modelMapper;
    }

    @Override
    public void postHandle(HttpServletRequest request,
                           HttpServletResponse response, Object handler,
                           ModelAndView modelAndView) {

        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();

        if (modelAndView != null && authentication != null) {
            Object principal = authentication.getPrincipal();

            if (!principal.equals("anonymousUser")) {
                CustomUserDetails customUserDetails = (CustomUserDetails) authentication.getPrincipal();

                UserDTO userDTO = modelMapper.map(customUserDetails.getUser(), UserDTO.class);

                modelAndView.getModel().put("user", userDTO);
            }
        }
    }
}