<#import "spring.ftl" as spring/>
<#import "base.ftlh" as base>

<@base.main [] "Регистрация">
    <section class="auth">
        <div class="auth__bg"></div>
        <div class="auth__inner">
            <a href="#" onclick="history.back();" class="btn__back">
                <svg class="arrow__left-symbol">
                    <use xlink:href="#arrow-left"></use>
                </svg>
            </a>
            <div class="auth__logo">
                <div class="auth__logo-inner">
                    <img src="/images/header/logo.png" alt="" class="auth__logo-img">
                </div>
            </div>
            <div class="auth__welcome">
                <div class="auth__welcome-title">Регистрация</div>
            </div>
            <@spring.bind "signUpForm"/>
            <form action="" class="login__form" method="post">
                <div class="input__group">
                    <@spring.formInput path="signUpForm.name" attributes="style='input__field' placeholder='Введите Ваше полное имя' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formInput path="signUpForm.email" attributes="style='input__field' placeholder='Введите адрес электронной почты' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formPasswordInput path="signUpForm.password" attributes="style='input__field' placeholder='Введите пароль' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formPasswordInput path="signUpForm.verifyPassword" attributes="style='input__field' placeholder='Введите пароль еще раз' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>

                <#if passwordMismatch??>
                    <div class="error">${passwordMismatch}</div>
                </#if>

                <button type=submit class="auth__registration-btn">Зарегистрироваться</button>
<#--                <a class="auth__social-google" href="/oauth2/authorization/google">-->
<#--                    <img class="social-google__icon" src="/images/auth/google-icon.png" alt="Google icon">-->
<#--                    <span class="social-google__text">Войти через Google</span>-->
<#--                </a>-->
            </form>
        </div>
    </section>
</@base.main>