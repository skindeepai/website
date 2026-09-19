// Main JavaScript for PLGL Website

// Mobile Menu Functions
const menuToggle = document.querySelector('.mobile-menu-toggle');
const mobileMenu = document.getElementById('mobileMenu');
if (menuToggle && mobileMenu) {
    menuToggle.setAttribute('aria-controls', 'mobileMenu');
    menuToggle.setAttribute('aria-expanded', 'false');
    menuToggle.setAttribute('aria-label', 'Open navigation');
    mobileMenu.addEventListener('click', function(event) {
        if (event.target.closest('a')) closeMobileMenu();
    });
}

function toggleMobileMenu() {
    if (!mobileMenu || !menuToggle) return;
    const open = mobileMenu.classList.toggle('active');
    menuToggle.setAttribute('aria-expanded', String(open));
    menuToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
}

function closeMobileMenu() {
    if (!mobileMenu || !menuToggle) return;
    mobileMenu.classList.remove('active');
    menuToggle.setAttribute('aria-expanded', 'false');
    menuToggle.setAttribute('aria-label', 'Open navigation');
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
    if (mobileMenu && menuToggle && !mobileMenu.contains(event.target) && !menuToggle.contains(event.target)) {
        closeMobileMenu();
    }
});

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && mobileMenu && mobileMenu.classList.contains('active')) {
        closeMobileMenu();
        menuToggle.focus();
    }
});
window.addEventListener('resize', function() {
    if (window.innerWidth > 900) closeMobileMenu();
});

// Tab functionality
function showTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');
    
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    buttons.forEach(button => {
        button.classList.remove('active');
    });
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const id = this.getAttribute('href').slice(1);
        const target = id ? document.getElementById(decodeURIComponent(id)) : null;
        if (target) {
            e.preventDefault();
            target.scrollIntoView({
                behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
                block: 'start'
            });
        }
    });
});

// Animated counter for hero stats
function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.01,
    rootMargin: '50px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('.section').forEach(section => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    section.style.opacity = '0';
    section.style.transform = 'translateY(20px)';
    section.style.transition = 'all 0.6s ease-out';
    observer.observe(section);
});

// Remove duplicate mobile menu code - using the functions defined at the top of this file

// Copy code functionality
document.querySelectorAll('pre').forEach(pre => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-wrapper';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);
    
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';
    wrapper.appendChild(button);
    
    button.addEventListener('click', () => {
        const code = (pre.querySelector('code') || pre).textContent;
        if (!navigator.clipboard) {
            button.textContent = 'Select text to copy';
            return;
        }
        navigator.clipboard.writeText(code).then(() => {
            button.textContent = 'Copied!';
            setTimeout(() => {
                button.textContent = 'Copy';
            }, 2000);
        }).catch(() => { button.textContent = 'Select text to copy'; });
    });
});

// Application filter
const categoryButtons = ['all', 'creative', 'science', 'product', 'content', 'social'];
const filterContainer = document.createElement('div');
filterContainer.className = 'filter-buttons';
filterContainer.innerHTML = categoryButtons.map(cat => 
    `<button class="filter-btn ${cat === 'all' ? 'active' : ''}" data-category="${cat}">
        ${cat.charAt(0).toUpperCase() + cat.slice(1)}
    </button>`
).join('');

const appGrid = document.querySelector('.applications-grid');
if (appGrid) {
    appGrid.parentNode.insertBefore(filterContainer, appGrid);
    
    filterContainer.addEventListener('click', (e) => {
        if (e.target.classList.contains('filter-btn')) {
            const category = e.target.dataset.category;
            
            // Update active button
            filterContainer.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            e.target.classList.add('active');
            
            // Filter cards
            document.querySelectorAll('.app-card').forEach(card => {
                if (category === 'all' || card.dataset.category === category) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    });
}

// Add styles for new elements (mobile menu styles should be in style.css)
const style = document.createElement('style');
style.textContent = `
    .code-wrapper {
        position: relative;
    }
    
    .copy-button {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        padding: 0.25rem 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 0.875rem;
    }
    
    .copy-button:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .filter-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
		flex-wrap: wrap;
    }
    
    .filter-btn {
        padding: 0.5rem 1.5rem;
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        border-radius: 2rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .filter-btn.active,
    .filter-btn:hover {
        background: var(--primary-color);
        color: white;
    }
    
    .section-dark .filter-btn {
        border-color: white;
        color: white;
    }
    
    .section-dark .filter-btn.active,
    .section-dark .filter-btn:hover {
        background: white;
        color: var(--dark-bg);
    }
`;
document.head.appendChild(style);
