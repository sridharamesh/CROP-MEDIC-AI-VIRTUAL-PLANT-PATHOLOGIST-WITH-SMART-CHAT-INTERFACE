// Mobile navigation toggle
document.addEventListener('DOMContentLoaded', function() {
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');
    
    // Toggle navigation
    if (burger) {
        burger.addEventListener('click', function() {
            nav.classList.toggle('nav-active');
            burger.classList.toggle('toggle');
        });
    }
    
    // Close navigation when clicking on a link
    const navLinks = document.querySelectorAll('.nav-links li');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            nav.classList.remove('nav-active');
            burger.classList.remove('toggle');
        });
    });
    
    // Added CSS for mobile nav when JavaScript is loaded
    const style = document.createElement('style');
    style.textContent = `
        @media (max-width: 768px) {
            .nav-links {
                position: absolute;
                right: 0;
                top: 70px;
                background-color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 50%;
                transform: translateX(100%);
                transition: transform 0.5s ease-in;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                z-index: 1;
            }
            
            .nav-active {
                transform: translateX(0%);
                display: flex;
            }
            
            .nav-links li {
                opacity: 0;
                margin: 0;
                width: 100%;
                text-align: center;
                padding: 15px 0;
            }
            
            .nav-active li {
                opacity: 1;
                animation: navLinkFade 0.5s ease forwards;
            }
            
            .burger.toggle .line1 {
                transform: rotate(-45deg) translate(-5px, 6px);
            }
            
            .burger.toggle .line2 {
                opacity: 0;
            }
            
            .burger.toggle .line3 {
                transform: rotate(45deg) translate(-5px, -6px);
            }
            
            @keyframes navLinkFade {
                from {
                    opacity: 0;
                    transform: translateX(50px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0px);
                }
            }
        }
    `;
    document.head.appendChild(style);
    
    // Initialize animations for nav links
    if (window.innerWidth <= 768) {
        navLinks.forEach((link, index) => {
            link.style.animation = `navLinkFade 0.5s ease forwards ${index / 7 + 0.3}s`;
        });
    }
});