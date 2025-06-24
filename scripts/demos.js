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

// 2D Latent Space Navigation Demo - Batch Generation & Refinement
const demo2DCanvas = document.getElementById('demo2D');
if (demo2DCanvas) {
    // Set canvas size responsively
    function resizeCanvas() {
        const container = demo2DCanvas.parentElement;
        const maxWidth = Math.min(600, container.clientWidth - 32); // 32px for padding
        demo2DCanvas.width = maxWidth;
        demo2DCanvas.height = Math.min(400, maxWidth * 0.67); // Maintain aspect ratio
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    const ctx = demo2DCanvas.getContext('2d');
    let animationId = null;
    let isRunning = false;
    
    // Demo state
    let currentBatch = 1;
    let allSamples = [];
    let currentBatchSamples = [];
    let classifier = null;
    let idealPoint = { x: 420, y: 180 }; // User's true preference (hidden)
    let uncertaintyMap = [];
    
    // Stats
    let totalSamples = 0;
    let accuracy = 0;
    
    // Color scheme
    const colors = {
        like: '#10B981',
        dislike: '#EF4444',
        exploration: '#8B5CF6',
        exploitation: '#3B82F6',
        ideal: '#F59E0B',
        uncertainty: '#6B7280'
    };
    
    function drawScene() {
        // Clear canvas
        ctx.fillStyle = '#FAFAFA';
        ctx.fillRect(0, 0, demo2DCanvas.width, demo2DCanvas.height);
        
        // Draw uncertainty/confidence heatmap
        if (classifier && uncertaintyMap.length > 0) {
            uncertaintyMap.forEach(point => {
                const confidence = getClassifierConfidence(point.x, point.y);
                const opacity = 0.15 * (1 - confidence); // Higher uncertainty = more visible
                ctx.fillStyle = `rgba(107, 114, 128, ${opacity})`;
                ctx.fillRect(point.x - 10, point.y - 10, 20, 20);
            });
        }
        
        // Draw all previous samples (faded)
        allSamples.forEach(sample => {
            if (!currentBatchSamples.includes(sample)) {
                ctx.globalAlpha = 0.3;
                drawSample(sample);
                ctx.globalAlpha = 1;
            }
        });
        
        // Draw current batch samples
        currentBatchSamples.forEach(sample => {
            drawSample(sample);
            
            // Highlight if it's exploration vs exploitation
            if (sample.type === 'exploration') {
                ctx.strokeStyle = colors.exploration;
                ctx.lineWidth = 3;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.arc(sample.x, sample.y, 25, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        });
        
        // Draw predicted high-preference regions
        if (classifier) {
            drawPreferenceRegions();
        }
        
        // Draw batch info
        ctx.fillStyle = '#1F2937';
        ctx.font = 'bold 16px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(`Batch ${currentBatch}: ${currentBatch === 1 ? 'Initial Exploration' : currentBatch === 2 ? 'Focused Refinement' : 'Fine-tuning + Exploration'}`, 20, 30);
        
        // Draw legend
        ctx.font = '12px Inter';
        ctx.fillStyle = '#6B7280';
        const legendY = 350;
        ctx.fillText('● Exploitation (refining known good areas)', 20, legendY - 20);
        ctx.fillText('◌ Exploration (checking uncertain areas)', 20, legendY );
        
        // Show accuracy improving
        if (accuracy > 0) {
            ctx.fillStyle = colors.like;
            ctx.font = 'bold 14px Inter';
            ctx.fillText(`Model Accuracy: ${Math.round(accuracy)}%`, 380, 20);
        }
    }
    
    function drawSample(sample) {
        // Draw influence gradient
        const gradient = ctx.createRadialGradient(sample.x, sample.y, 0, sample.x, sample.y, 50);
        if (sample.rated) {
            if (sample.liked) {
                gradient.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
                gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');
            } else {
                gradient.addColorStop(0, 'rgba(239, 68, 68, 0.2)');
                gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
            }
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(sample.x, sample.y, 50, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // Draw sample point
        ctx.beginPath();
        ctx.arc(sample.x, sample.y, 12, 0, Math.PI * 2);
        if (sample.rated) {
            ctx.fillStyle = sample.liked ? colors.like : colors.dislike;
        } else {
            ctx.fillStyle = '#E5E7EB';
        }
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw rating emoji
        if (sample.rated) {
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(sample.liked ? '👍' : '👎', sample.x, sample.y);
        }
    }
    
    function drawPreferenceRegions() {
        // Draw contour lines for preference predictions
        ctx.strokeStyle = colors.exploitation;
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 5]);
        
        // Simple visualization of high-preference regions
        const regions = getHighPreferenceRegions();
        regions.forEach(region => {
            ctx.globalAlpha = 0.1;
            ctx.fillStyle = colors.exploitation;
            ctx.beginPath();
            ctx.arc(region.x, region.y, region.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1;
        });
        
        ctx.setLineDash([]);
    }
    
    function getUserPreference(x, y) {
        // Simulate user's hidden preference function with multiple peaks
        const dist1 = Math.sqrt((x - idealPoint.x) ** 2 + (y - idealPoint.y) ** 2);
        const dist2 = Math.sqrt((x - 150) ** 2 + (y - 250) ** 2);
        
        // Main preference peak + secondary peak
        const preference = Math.exp(-dist1 / 80) + 0.3 * Math.exp(-dist2 / 100);
        
        // Add noise and threshold
        return preference > 0.45 + (Math.random() - 0.5) * 0.2;
    }
    
    function generateBatch(batchNumber) {
        const batchSize = 8;
        const samples = [];
        
        if (batchNumber === 1) {
            // First batch: Pure exploration - grid sampling
            for (let i = 0; i < batchSize; i++) {
                const angle = (i / batchSize) * Math.PI * 2;
                const radius = 120 + (i % 2) * 80;
                samples.push({
                    x: demo2DCanvas.width / 2 + Math.cos(angle) * radius,
                    y: demo2DCanvas.height / 2 + Math.sin(angle) * radius,
                    type: 'exploration',
                    rated: false,
                    batch: batchNumber
                });
            }
        } else {
            // Later batches: Mix of exploitation and exploration
            const exploitCount = Math.floor(batchSize * 0.7); // 70% exploitation
            const exploreCount = batchSize - exploitCount; // 30% exploration
            
            // Exploitation: Generate near high-confidence positive areas
            const goodRegions = getHighPreferenceRegions();
            for (let i = 0; i < exploitCount && goodRegions.length > 0; i++) {
                const region = goodRegions[i % goodRegions.length];
                const angle = Math.random() * Math.PI * 2;
                const r = Math.random() * region.radius * 0.8;
                samples.push({
                    x: region.x + Math.cos(angle) * r,
                    y: region.y + Math.sin(angle) * r,
                    type: 'exploitation',
                    rated: false,
                    batch: batchNumber
                });
            }
            
            // Exploration: Sample uncertain areas
            for (let i = 0; i < exploreCount; i++) {
                const uncertainPoint = getUncertainArea();
                samples.push({
                    x: uncertainPoint.x,
                    y: uncertainPoint.y,
                    type: 'exploration',
                    rated: false,
                    batch: batchNumber
                });
            }
        }
        
        // Keep samples in bounds
        samples.forEach(s => {
            s.x = Math.max(30, Math.min(demo2DCanvas.width - 30, s.x));
            s.y = Math.max(30, Math.min(demo2DCanvas.height - 30, s.y));
        });
        
        return samples;
    }
    
    function getHighPreferenceRegions() {
        if (!classifier || allSamples.length < 5) return [];
        
        const regions = [];
        const likedSamples = allSamples.filter(s => s.rated && s.liked);
        
        // Cluster liked samples
        likedSamples.forEach(sample => {
            let merged = false;
            for (let region of regions) {
                const dist = Math.sqrt((region.x - sample.x) ** 2 + (region.y - sample.y) ** 2);
                if (dist < 100) {
                    // Merge into existing region
                    region.x = (region.x * region.count + sample.x) / (region.count + 1);
                    region.y = (region.y * region.count + sample.y) / (region.count + 1);
                    region.count++;
                    region.radius = Math.min(80, 30 + region.count * 10);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                regions.push({
                    x: sample.x,
                    y: sample.y,
                    radius: 40,
                    count: 1
                });
            }
        });
        
        return regions.sort((a, b) => b.count - a.count);
    }
    
    function getUncertainArea() {
        // Find areas far from existing samples
        let bestX = 0, bestY = 0, maxMinDist = 0;
        
        for (let i = 0; i < 20; i++) {
            const x = Math.random() * (demo2DCanvas.width - 60) + 30;
            const y = Math.random() * (demo2DCanvas.height - 60) + 30;
            
            let minDist = Infinity;
            allSamples.forEach(sample => {
                const dist = Math.sqrt((sample.x - x) ** 2 + (sample.y - y) ** 2);
                minDist = Math.min(minDist, dist);
            });
            
            if (minDist > maxMinDist) {
                maxMinDist = minDist;
                bestX = x;
                bestY = y;
            }
        }
        
        return { x: bestX, y: bestY };
    }
    
    function getClassifierConfidence(x, y) {
        if (!classifier || allSamples.length === 0) return 0;
        
        // Simple confidence based on distance to nearest samples
        let nearestDist = Infinity;
        allSamples.forEach(sample => {
            if (sample.rated) {
                const dist = Math.sqrt((sample.x - x) ** 2 + (sample.y - y) ** 2);
                nearestDist = Math.min(nearestDist, dist);
            }
        });
        
        return Math.max(0, 1 - nearestDist / 150);
    }
    
    function updateClassifier() {
        // Simple classifier based on rated samples
        classifier = {
            predict: function(x, y) {
                let weightedSum = 0;
                let totalWeight = 0;
                
                allSamples.forEach(sample => {
                    if (sample.rated) {
                        const dist = Math.sqrt((sample.x - x) ** 2 + (sample.y - y) ** 2) + 1;
                        const weight = 1 / (dist * dist);
                        weightedSum += (sample.liked ? 1 : -1) * weight;
                        totalWeight += weight;
                    }
                });
                
                return totalWeight > 0 ? weightedSum / totalWeight : 0;
            }
        };
        
        // Calculate accuracy
        let correct = 0;
        let total = 0;
        allSamples.forEach(sample => {
            if (sample.rated) {
                const predicted = classifier.predict(sample.x, sample.y) > 0;
                if (predicted === sample.liked) correct++;
                total++;
            }
        });
        accuracy = total > 0 ? (correct / total) * 100 : 0;
    }
    
    let frameCount = 0;
    let ratingDelay = 30; // frames between ratings
    let batchDelay = 120; // frames between batches
    let lastRatingFrame = 0;
    let waitingForNextBatch = false;
    let batchCompleteFrame = 0;
    
    function animate() {
        if (!isRunning) return;
        
        frameCount++;
        
        // Rate samples one by one with delay
        if (!waitingForNextBatch && frameCount - lastRatingFrame > ratingDelay) {
            let unratedSample = currentBatchSamples.find(s => !s.rated);
            if (unratedSample) {
                unratedSample.liked = getUserPreference(unratedSample.x, unratedSample.y);
                unratedSample.rated = true;
                totalSamples++;
                lastRatingFrame = frameCount;
                
                // Update stats
                updateStats();
                
                // Check if batch is complete
                if (currentBatchSamples.every(s => s.rated)) {
                    waitingForNextBatch = true;
                    batchCompleteFrame = frameCount;
                    updateClassifier();
                }
            }
        }
        
        // Wait before generating next batch
        if (waitingForNextBatch && frameCount - batchCompleteFrame > batchDelay) {
            if (currentBatch < 3) {
                // Generate next batch
                currentBatch++;
                currentBatchSamples = generateBatch(currentBatch);
                currentBatchSamples.forEach(s => allSamples.push(s));
                waitingForNextBatch = false;
                updateStats();
            } else {
                // Demo complete
                isRunning = false;
                showFinalResult();
            }
        }
        
        drawScene();
        
        if (isRunning) {
            animationId = requestAnimationFrame(animate);
        }
    }
    
    function updateStats() {
        document.getElementById('batch-number').textContent = currentBatch;
        document.getElementById('sample-count').textContent = totalSamples;
        document.getElementById('accuracy').textContent = Math.round(accuracy) + '%';
    }
    
    function showFinalResult() {
        // Highlight the best found region
        const regions = getHighPreferenceRegions();
        if (regions.length > 0) {
            ctx.save();
            ctx.strokeStyle = colors.ideal;
            ctx.lineWidth = 3;
            ctx.setLineDash([10, 5]);
            ctx.beginPath();
            ctx.arc(regions[0].x, regions[0].y, regions[0].radius + 10, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
            
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('⭐', regions[0].x, regions[0].y);
            
            ctx.fillStyle = '#1F2937';
            ctx.font = 'bold 14px Inter';
            ctx.fillText('Ideal region found!', regions[0].x, regions[0].y + 40);
        }
    }
    
    // Initialize uncertainty map
    function initUncertaintyMap() {
        uncertaintyMap = [];
        for (let x = 20; x < demo2DCanvas.width; x += 20) {
            for (let y = 20; y < demo2DCanvas.height; y += 20) {
                uncertaintyMap.push({ x, y });
            }
        }
    }
    
    window.startDemo2D = function() {
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        
        // Reset state
        isRunning = true;
        currentBatch = 1;
        allSamples = [];
        totalSamples = 0;
        accuracy = 0;
        classifier = null;
        
        // Reset animation timing
        frameCount = 0;
        lastRatingFrame = 0;
        waitingForNextBatch = false;
        batchCompleteFrame = 0;
        
        // Initialize uncertainty map
        initUncertaintyMap();
        
        // Generate first batch
        currentBatchSamples = generateBatch(1);
        currentBatchSamples.forEach(s => allSamples.push(s));
        
        // Start animation
        updateStats();
        animate();
    };
    
    window.resetDemo2D = function() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        isRunning = false;
        currentBatch = 1;
        allSamples = [];
        currentBatchSamples = [];
        totalSamples = 0;
        accuracy = 0;
        classifier = null;
        
        updateStats();
        drawScene();
    };
    
    // Initialize
    initUncertaintyMap();
    drawScene();
}

// Preference Learning Simulation
const sampleGrid = document.getElementById('sampleGrid');
const learningCurveCanvas = document.getElementById('learningCurve');

if (sampleGrid && learningCurveCanvas) {
    // Set canvas size responsively
    function resizeLearningCanvas() {
        const container = learningCurveCanvas.parentElement;
        const maxWidth = Math.min(400, container.clientWidth - 32);
        learningCurveCanvas.width = maxWidth;
        learningCurveCanvas.height = Math.min(200, maxWidth * 0.5);
    }
    
    resizeLearningCanvas();
    window.addEventListener('resize', resizeLearningCanvas);
    
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