<#import "spring.ftl" as spring/>
<#import "base.ftlh" as base>

<@base.main [] "Форма для заполнения анализов">
    <section class="auth">
        <div class="auth__bg"></div>
        <div class="auth__inner">
            <div class="auth__logo">
                <div class="auth__logo-inner">
                    <img src="/images/header/logo.png" alt="" class="auth__logo-img">
                </div>
            </div>
            <div class="auth__welcome">
                <div class="auth__welcome-title">Форма для заполнения анализов</div>
            </div>
            <@spring.bind "userAnalyzeForm"/>
            <form action="/analyze/users/<#if user??>${user.id}</#if>" class="login__form" method="post">
                <div class="auth__welcome-title1">Вы первый раз проходите обследование?</div>
                <div class="input__group1">
                    <label for="option1" class="label__group">Да</label><br>
                    <input class="input__field1" type="radio" id="option1" name="primarySecondaryCvrm" value="Primary">
                </div>
                <div class="input__group1">
                    <label for="option2" class="label__group">Нет</label><br>
                    <input class="input__field1" type="radio" id="option2" name="primarySecondaryCvrm"
                           value="Secondary">
                </div>

                <div class="auth__welcome-title1">У вас есть гипертония?</div>
                <div class="input__group1">
                    <label for="option1" class="label__group">Да</label><br>
                    <input class="input__field1" type="radio" id="option1" name="hypertension" value="Yes">
                </div>
                <div class="input__group1">
                    <label for="option2" class="label__group">Нет</label><br>
                    <input class="input__field1" type="radio" id="option2" name="hypertension"
                           value="No">
                </div>

                <div class="auth__welcome-title1">Выберите ваш пол</div>
                <div class="input__group1">
                    <label for="option1" class="label__group">Мужчина</label><br>
                    <input class="input__field1" type="radio" id="option1" name="gender" value="M">
                </div>
                <div class="input__group1">
                    <label for="option2" class="label__group">Женщина</label><br>
                    <input class="input__field1" type="radio" id="option2" name="gender"
                           value="F">
                </div>


                <div class="auth__welcome-title1">Вы курите?</div>
                <div class="input__group1">
                    <label for="option1" class="label__group">Да</label><br>
                    <input class="input__field1" type="radio" id="option1" name="smokingStatus" value="Yes">
                </div>
                <div class="input__group1">
                    <label for="option2" class="label__group">Нет</label><br>
                    <input class="input__field1" type="radio" id="option2" name="smokingStatus"
                           value="No">
                </div>
                <div class="input__group1">
                    <label for="option2" class="label__group">Раньше курил</label><br>
                    <input class="input__field1" type="radio" id="option2" name="smokingStatus"
                           value="Previously">
                </div>

                <div class="input__group">
                    <@spring.formInput path="userAnalyzeForm.organisationName" attributes="style='input__field' placeholder='К какой поликлините вы прикреплены?' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formInput path="userAnalyzeForm.glucoseFasting" attributes="style='input__field' placeholder='Введите ваш уровень глюкозы в крови' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formInput path="userAnalyzeForm.systolicBloodPressure" attributes="style='input__field' placeholder='Введите ваше систолическое давление' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formInput path="userAnalyzeForm.diastolicBloodPressure" attributes="style='input__field' placeholder='Введите ваше диастоличсекое давление' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formInput path="userAnalyzeForm.bmi" attributes="style='input__field' placeholder='Введите ваш идекс тела' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <div class="input__group">
                    <@spring.formInput path="userAnalyzeForm.age" attributes="style='input__field' placeholder='Введите ваш возраст' class='input__field'"/>
                    <@spring.showErrors separator="<br>" classOrStyle="error"/>
                </div>
                <button type=submit class="auth__registration-btn">Отправить анализы</button>
                <#--                <a class="auth__social-google" href="/oauth2/authorization/google">-->
                <#--                    <img class="social-google__icon" src="/images/auth/google-icon.png" alt="Google icon">-->
                <#--                    <span class="social-google__text">Войти через Google</span>-->
                <#--                </a>-->
            </form>
        </div>
    </section>
</@base.main>