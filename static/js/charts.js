// Chart.js configurations and initialization
function initializeCharts(predictionData) {
    // Risk Gauge Chart
    initializeRiskGauge(predictionData);
    
    // Risk Factors Chart
    initializeRiskFactorsChart(predictionData);
    
    // Feature Importance Chart
    initializeFeatureImportanceChart(predictionData);
}

function initializeRiskGauge(data) {
    const ctx = document.getElementById('riskGauge');
    if (!ctx) return;
    
    const riskPercentage = data.risk_percentage;
    const remainingPercentage = 100 - riskPercentage;
    
    // Determine color based on risk level
    let riskColor;
    if (riskPercentage < 30) {
        riskColor = '#28a745'; // Green
    } else if (riskPercentage < 70) {
        riskColor = '#ffc107'; // Yellow
    } else {
        riskColor = '#dc3545'; // Red
    }
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Cancer Risk', 'No Risk'],
            datasets: [{
                data: [riskPercentage, remainingPercentage],
                backgroundColor: [riskColor, '#e9ecef'],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(1) + '%';
                        }
                    }
                }
            },
            elements: {
                arc: {
                    borderWidth: 0
                }
            }
        },
        plugins: [{
            beforeDraw: function(chart) {
                const width = chart.width;
                const height = chart.height;
                const ctx = chart.ctx;
                
                ctx.restore();
                const fontSize = (height / 114).toFixed(2);
                ctx.font = fontSize + "em Arial";
                ctx.textBaseline = "middle";
                ctx.fillStyle = riskColor;
                
                const text = riskPercentage.toFixed(1) + "%";
                const textX = Math.round((width - ctx.measureText(text).width) / 2);
                const textY = height / 2;
                
                ctx.fillText(text, textX, textY);
                ctx.save();
            }
        }]
    });
}

function initializeRiskFactorsChart(data) {
    const ctx = document.getElementById('riskFactorsChart');
    if (!ctx) return;
    
    const riskFactors = data.risk_factors;
    const labels = [];
    const scores = [];
    const colors = [];
    
    // Process risk factors data
    Object.keys(riskFactors).forEach(factor => {
        const factorData = riskFactors[factor];
        labels.push(factor.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()));
        scores.push(factorData.score);
        
        // Color based on risk level
        if (factorData.level === 'Low') {
            colors.push('#28a745');
        } else if (factorData.level === 'Moderate') {
            colors.push('#ffc107');
        } else {
            colors.push('#dc3545');
        }
    });
    
    new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Score',
                data: scores,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const factor = Object.keys(riskFactors)[context.dataIndex];
                            const factorData = riskFactors[factor];
                            return [
                                `Risk Level: ${factorData.level}`,
                                `Score: ${factorData.score}/3`,
                                `${factorData.description}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 3,
                    ticks: {
                        stepSize: 1
                    },
                    title: {
                        display: true,
                        text: 'Risk Score (1-3)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Risk Factors'
                    }
                }
            }
        }
    });
}

function initializeFeatureImportanceChart(data) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;
    
    const featureImportance = data.feature_importance;
    const labels = Object.keys(featureImportance);
    const values = Object.values(featureImportance);
    
    // Create gradient colors
    const colors = labels.map((_, index) => {
        const hue = (index * 360 / labels.length) % 360;
        return `hsl(${hue}, 70%, 50%)`;
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Importance',
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${(context.parsed.y * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Features'
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Utility functions for chart animations
function animateNumber(element, start, end, duration = 1000) {
    const startTime = performance.now();
    const change = end - start;
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = start + (change * easeOutCubic(progress));
        element.textContent = current.toFixed(1);
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add loading states to forms
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function() {
            // Add loading spinner to submit button
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
                submitBtn.disabled = true;
            }
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Chart.js global configuration
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.color = '#6c757d';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
Chart.defaults.plugins.tooltip.titleColor = '#fff';
Chart.defaults.plugins.tooltip.bodyColor = '#fff';
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.displayColors = false;

// Responsive chart configuration
function makeChartResponsive() {
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
}

// Initialize responsive charts
makeChartResponsive();
