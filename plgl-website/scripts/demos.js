// Interactive Demos for PLGL Website

// Latent Space Visualization (Hero Section)
const latentCanvas = document.getElementById('latentSpaceVisualization');
if (latentCanvas) {
    const ctx = latentCanvas.getContext('2d');
    const particles = [];
    const particleCount = 50;
    
    // Set canvas size
    function resizeCanvas() {
        latentCanvas.width = latentCanvas.offsetWidth;
        latentCanvas.height = latentCanvas.offsetHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Particle class
    class Particle {
        constructor() {
            this.x = Math.random() * latentCanvas.width;
            this.y = Math.random() * latentCanvas.height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 3 + 2;
            this.preference = Math.random();
            this.hue = this.preference * 60 + 240; // Blue to purple gradient
        }
        
        update() {
            this.x += this.vx;
            this.y += this.vy;
            
            // Bounce off walls
            if (this.x < 0 || this.x > latentCanvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > latentCanvas.height) this.vy *= -1;
            
            // Constrain to canvas
            this.x = Math.max(0, Math.min(latentCanvas.width, this.x));
            this.y = Math.max(0, Math.min(latentCanvas.height, this.y));
        }
        
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue}, 70%, 60%, ${0.3 + this.preference * 0.7})`;
            ctx.fill();
            
            // Glow effect
            const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 3);
            gradient.addColorStop(0, `hsla(${this.hue}, 70%, 60%, ${0.2 * this.preference})`);
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.fill();
        }
    }
    
    // Initialize particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
    
    // Animation loop
    function animateLatentSpace() {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(0, 0, latentCanvas.width, latentCanvas.height);
        
        // Draw connections
        particles.forEach((p1, i) => {
            particles.slice(i + 1).forEach(p2 => {
                const distance = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
                if (distance < 100) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 * (1 - distance / 100)})`;
                    ctx.stroke();
                }
            });
        });
        
        // Update and draw particles
        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });
        
        requestAnimationFrame(animateLatentSpace);
    }
    
    animateLatentSpace();
}

// 2D Latent Space Navigation Demo
const demo2DCanvas = document.getElementById('demo2D');
if (demo2DCanvas) {
    const ctx = demo2DCanvas.getContext('2d');
    let isRunning = false;
    let currentPoint = null;
    let targetPoint = null;
    let preferenceMap = [];
    let trajectory = [];
    
    // Initialize preference map (simulated)
    function initPreferenceMap() {
        preferenceMap = [];
        for (let x = 0; x < demo2DCanvas.width; x += 20) {
            for (let y = 0; y < demo2DCanvas.height; y += 20) {
                // Create multiple preference peaks
                const peak1 = Math.exp(-((x - 150) ** 2 + (y - 150) ** 2) / 5000);
                const peak2 = Math.exp(-((x - 450) ** 2 + (y - 250) ** 2) / 5000);
                const peak3 = Math.exp(-((x - 300) ** 2 + (y - 100) ** 2) / 3000);
                const preference = Math.max(peak1, peak2, peak3) + Math.random() * 0.1;
                
                preferenceMap.push({
                    x: x,
                    y: y,
                    preference: Math.min(1, preference)
                });
            }
        }
    }
    
    function drawPreferenceMap() {
        preferenceMap.forEach(point => {
            const hue = point.preference * 120; // Red to green
            ctx.fillStyle = `hsla(${hue}, 70%, 50%, 0.5)`;
            ctx.fillRect(point.x - 10, point.y - 10, 20, 20);
        });
    }
    
    function getPreferenceAt(x, y) {
        // Interpolate preference value at arbitrary point
        let closestPoint = preferenceMap[0];
        let minDist = Infinity;
        
        preferenceMap.forEach(point => {
            const dist = Math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2);
            if (dist < minDist) {
                minDist = dist;
                closestPoint = point;
            }
        });
        
        return closestPoint.preference;
    }
    
    function gradientAscent() {
        if (!currentPoint || !isRunning) return;
        
        // Calculate gradient
        const step = 5;
        const currentPref = getPreferenceAt(currentPoint.x, currentPoint.y);
        const rightPref = getPreferenceAt(currentPoint.x + step, currentPoint.y);
        const topPref = getPreferenceAt(currentPoint.x, currentPoint.y - step);
        
        const gradX = (rightPref - currentPref) / step;
        const gradY = (topPref - currentPref) / step;
        
        // Update position
        const learningRate = 10;
        currentPoint.x += gradX * learningRate;
        currentPoint.y += gradY * learningRate;
        
        // Add to trajectory
        trajectory.push({x: currentPoint.x, y: currentPoint.y});
        
        // Redraw
        ctx.clearRect(0, 0, demo2DCanvas.width, demo2DCanvas.height);
        drawPreferenceMap();
        
        // Draw trajectory
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.beginPath();
        trajectory.forEach((point, i) => {
            if (i === 0) ctx.moveTo(point.x, point.y);
            else ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
        
        // Draw current point
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(currentPoint.x, currentPoint.y, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Continue optimization
        if (isRunning) {
            requestAnimationFrame(gradientAscent);
        }
    }
    
    window.startDemo2D = function() {
        isRunning = true;
        currentPoint = {
            x: Math.random() * demo2DCanvas.width,
            y: Math.random() * demo2DCanvas.height
        };
        trajectory = [currentPoint];
        gradientAscent();
    };
    
    window.resetDemo2D = function() {
        isRunning = false;
        trajectory = [];
        ctx.clearRect(0, 0, demo2DCanvas.width, demo2DCanvas.height);
        drawPreferenceMap();
    };
    
    // Initialize
    initPreferenceMap();
    drawPreferenceMap();
}

// Preference Learning Simulation
const sampleGrid = document.getElementById('sampleGrid');
const learningCurveCanvas = document.getElementById('learningCurve');

if (sampleGrid && learningCurveCanvas) {
    let samples = [];
    let ratings = [];
    let accuracy = [];
    
    // Generate random samples (simulated faces/content)
    function generateSamples() {
        samples = [];
        sampleGrid.innerHTML = '';
        
        for (let i = 0; i < 8; i++) {
            const sample = {
                id: i,
                features: [Math.random(), Math.random(), Math.random()],
                element: null
            };
            
            // Create visual representation
            const div = document.createElement('div');
            div.className = 'sample-item';
            div.style.cssText = `
                width: 100px;
                height: 100px;
                border-radius: 10px;
                cursor: pointer;
                position: relative;
                overflow: hidden;
                background: linear-gradient(135deg, 
                    hsl(${sample.features[0] * 360}, 70%, 70%) 0%, 
                    hsl(${sample.features[1] * 360}, 70%, 70%) 100%);
                transition: all 0.3s;
            `;
            
            // Add click handler for rating
            div.addEventListener('click', () => rateSample(sample.id));
            
            sample.element = div;
            samples.push(sample);
            sampleGrid.appendChild(div);
        }
    }
    
    function rateSample(id) {
        const sample = samples[id];
        const rating = !sample.rated;
        
        sample.rated = rating;
        ratings.push({features: sample.features, rating: rating});
        
        // Visual feedback
        sample.element.style.border = rating ? '3px solid #10B981' : '3px solid #EF4444';
        sample.element.innerHTML = rating ? 
            '<div style="position: absolute; top: 5px; right: 5px; font-size: 20px;">👍</div>' : 
            '<div style="position: absolute; top: 5px; right: 5px; font-size: 20px;">👎</div>';
        
        // Update learning curve
        updateLearningCurve();
        
        // Generate new sample to replace rated one
        if (ratings.length < 20) {
            setTimeout(() => {
                const newSample = {
                    id: id,
                    features: [Math.random(), Math.random(), Math.random()],
                    element: sample.element
                };
                
                newSample.element.style.background = `linear-gradient(135deg, 
                    hsl(${newSample.features[0] * 360}, 70%, 70%) 0%, 
                    hsl(${newSample.features[1] * 360}, 70%, 70%) 100%)`;
                newSample.element.style.border = 'none';
                newSample.element.innerHTML = '';
                
                samples[id] = newSample;
            }, 500);
        }
    }
    
    function updateLearningCurve() {
        // Simulate improving accuracy
        const currentAccuracy = 0.5 + (ratings.length / 40) * 0.4 + Math.random() * 0.1;
        accuracy.push(Math.min(0.95, currentAccuracy));
        
        // Draw learning curve
        const ctx = learningCurveCanvas.getContext('2d');
        ctx.clearRect(0, 0, learningCurveCanvas.width, learningCurveCanvas.height);
        
        // Draw axes
        ctx.strokeStyle = '#E5E7EB';
        ctx.beginPath();
        ctx.moveTo(40, 10);
        ctx.lineTo(40, 170);
        ctx.lineTo(380, 170);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = '#6B7280';
        ctx.font = '12px Inter';
        ctx.fillText('100%', 5, 15);
        ctx.fillText('50%', 10, 95);
        ctx.fillText('0%', 15, 175);
        ctx.fillText('Ratings', 180, 190);
        
        // Draw curve
        ctx.strokeStyle = '#6366F1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        accuracy.forEach((acc, i) => {
            const x = 40 + (i / 20) * 340;
            const y = 170 - acc * 160;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        // Current accuracy text
        if (accuracy.length > 0) {
            const latestAcc = accuracy[accuracy.length - 1];
            ctx.fillStyle = '#6366F1';
            ctx.font = 'bold 14px Inter';
            ctx.fillText(`Preference Accuracy: ${(latestAcc * 100).toFixed(1)}%`, 100, 30);
        }
    }
    
    // Initialize
    generateSamples();
    
    // Add reset button
    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset Simulation';
    resetBtn.style.cssText = `
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background: #6366F1;
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        font-weight: 600;
    `;
    resetBtn.addEventListener('click', () => {
        ratings = [];
        accuracy = [];
        generateSamples();
        updateLearningCurve();
    });
    
    document.getElementById('preferenceDemo').appendChild(resetBtn);
}