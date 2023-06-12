<#import "base.ftlh" as base>

<@base.main [] "О нас">
    <section class="contact--title">
        <div class="about__bg"></div>
        <div class="contact--title-inner">
            <div class="contact__inner-title">Про нас</div>
            <div class="contact__inner-subtitle">
                <a href="/">Домашняя</a>
                <div class="contact__line">/</div>
                <div>О нас</div>
            </div>
        </div>
    </section>

    <section class="contacts">
        <div class="contacts__inner">
            <div class="about__mission">
                <div class="about__photo">
                    <div class="about__img">
<#--                        <img src="/images/about.png">-->
                    </div>
                </div>
                <div class="about__idea">
                    <div class="about__suptitle">
                        Внимание к здоровью есть наибольшее препятствие для жизни. – Платон
                    </div>
                    <div class="about__title">
                        Наша миссия
                    </div>
                    <div class="about__text">
                        Основная часть медицинских информационных систем основанных на машинном обучении направлены на
                        предсказание болезней у
                        пациентов, способах и методах их лечения. В этих системах предполагается, что у всех
                        потенциальных клиентов уже существуют подозрения или они
                        знают о каких-то проблемах в здоровье, но чаще всего люди не подозревают о своем диагнозе, что
                        приводит к печальным событиям. Мы же решает такого рода проблемы, потому что
                        наше приложение являться, так называемым индикатором возможных проблем со здоровьем или необходимости
                        обратиться к врачу
                    </div>
                </div>
            </div>
        </div>
    </section>

    <footer class="footer">
        <div class="footer__inner">
            <div class="footer__logo">
                <img src="/images/header/logo.png" alt="" class="footer__logo-img">
            </div>
            <div class="footer__copyright">
                © 2023. Все права защищены.
            </div>
        </div>
    </footer>

    <script src="/js/header.js"></script>
</@base.main>