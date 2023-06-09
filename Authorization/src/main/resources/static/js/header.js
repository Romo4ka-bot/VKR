const mainPage = document.querySelector('#page');
const menu = document.querySelector(".burger");
const closeBtn = document.querySelector(".close__block");
const links = document.querySelector("#links");
const allLinks = document.querySelectorAll(".nav__link-item");
const header = document.querySelector('#header');
const btnProfile = document.querySelector('.header__profile--svg');
const profileContent = document.querySelector('.header__profile-content');

const allFilterMovies = document.querySelectorAll(".movies__filter-item");
const openMenu = () => {
    if (links.className.includes("open")) {
        links.classList.remove('open');
        allLinks.forEach((item) => {
            item.classList.remove("fade");
        });
        mainPage.classList.remove("overflow-hidden");
    } else {
        links.classList.add('open');
        allLinks.forEach((item) => {
            item.classList.add("fade");
        });
        mainPage.classList.add("overflow-hidden");
    }
};

const closeMenu = () => {
    if (links.className.includes("open")) {
        links.classList.remove('open');
        allLinks.forEach((item) => {
            item.classList.remove("fade");
        });
        mainPage.classList.remove("overflow-hidden");
    }
};

const tooltips = () => {
    profileContent.classList.toggle('vis');
};


menu.addEventListener("click", openMenu);
closeBtn.addEventListener("click", closeMenu);
allLinks.forEach((item) => {
    item.addEventListener("click", closeMenu);
});
allFilterMovies.forEach((item) => {
    item.addEventListener("click", function (e) {
        e.preventDefault();
        if (!e.target.getAttribute('class').includes('filter-active')) {
            allFilterMovies.forEach((i) => {
                i.classList.remove('filter-active');
            });
            this.classList.add('filter-active');
        }
    });
});
btnProfile.addEventListener("click", tooltips);