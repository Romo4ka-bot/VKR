<#import "base.ftlh" as base>

<@base.main [] "Главная страница">
        <section class="popular">
            <div class="popular__inner">
                <div class="popular__img">
                    <div class="img-popular" style="background: url('/images/default_cover.png') no-repeat center;background-size: cover;"></div>
                </div>
                <h2 class="popular__title">MyCholesterol</h2>
                <p class="populat__description-short">Следите за своим здоровьем с нами</p>
            </div>
        </section>


    <section class="new__movies">
        <div class="new__movies-inner">
            <div class="new__movies-header">
<#--                <h2 class="new__movies-title">Здесь вы можете в реальном времени проверить свой уровень холестерина</h2>-->
                <p class="new__movies-subtitle">Здесь вы можете в реальном времени проверить свой уровень холестерина</p>
            </div>
            <div class="new__movies-body">
                <div class="view__more-block">
                    <a href="/health?predict=0" class="view__more-btn">Проверить мое здоровье</a>
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