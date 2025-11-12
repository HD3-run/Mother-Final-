// Enhanced analytics dashboard for MotherX AI
class MotherXAnalytics {
    constructor() {
        this.charts = {};
        this.analyticsData = {};
        this.updateInterval = null;
        this.refreshRate = 60000; // 1 minute
        
        this.chartConfigs = {
            emotionalTrends: {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                color: '#fff',
                                font: {
                                    size: 12
                                }
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            borderColor: '#007bff',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#adb5bd',
                                font: {
                                    size: 11
                                }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                color: '#adb5bd',
                                font: {
                                    size: 11
                                },
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    },
                    elements: {
                        line: {
                            tension: 0.4,
                            borderWidth: 2
                        },
                        point: {
                            radius: 4,
                            hoverRadius: 6
                        }
                    }
                }
            },
            interactionTypes: {
                type: 'doughnut',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#fff',
                                font: {
                                    size: 11
                                },
                                padding: 15,
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            borderColor: '#007bff',
                            borderWidth: 1,
                            callbacks: {
                                label: function(context) {
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((context.parsed / total) * 100).toFixed(1);
                                    return `${context.label}: ${percentage}%`;
                                }
                            }
                        }
                    },
                    cutout: '60%',
                    elements: {
                        arc: {
                            borderWidth: 2,
                            borderColor: '#212529'
                        }
                    }
                }
            }
        };
        
        this.colorPalette = {
            primary: '#0d6efd',
            secondary: '#6c757d',
            success: '#198754',
            danger: '#dc3545',
            warning: '#ffc107',
            info: '#0dcaf0',
            light: '#f8f9fa',
            dark: '#212529',
            emotions: {
                happy: '#28a745',
                sad: '#17a2b8',
                angry: '#dc3545',
                excited: '#ffc107',
                worried: '#6c757d',
                neutral: '#6f42c1',
                fear: '#fd7e14',
                surprise: '#e83e8c'
            },
            interactions: {
                question: '#0d6efd',
                support_request: '#dc3545',
                emotional_expression: '#e83e8c',
                sharing: '#20c997',
                casual_chat: '#6f42c1',
                reflection_request: '#fd7e14',
                goal_discussion: '#198754',
                problem_solving: '#ffc107',
                learning_request: '#17a2b8'
            }
        };
    }
    
    async initialize() {
        try {
            await this.loadAnalyticsData();
            this.initializeCharts();
            this.updatePredictiveInsights();
            this.setupTabListeners();
            this.startAutoRefresh();
            
            console.log('MotherX Analytics initialized');
        } catch (error) {
            console.error('Analytics initialization failed:', error);
            this.showError('Failed to initialize analytics dashboard');
        }
    }
    
    async loadAnalyticsData() {
        try {
            const response = await fetch('/analytics');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.analyticsData = await response.json();
            
            if (this.analyticsData.error) {
                throw new Error(this.analyticsData.error);
            }
            
        } catch (error) {
            console.error('Analytics data loading failed:', error);
            // Set default empty data structure
            this.analyticsData = {
                usage_stats: {
                    interaction_types: {},
                    emotional_states: {}
                },
                sentiment_trends: [],
                interaction_patterns: {},
                predictions: {
                    available_models: [],
                    model_status: {}
                },
                identity_evolution: []
            };
            throw error;
        }
    }
    
    initializeCharts() {
        this.initializeEmotionalTrendsChart();
        this.initializeInteractionTypesChart();
    }
    
    initializeEmotionalTrendsChart() {
        const canvas = document.getElementById('emotionalTrendsChart');
        if (!canvas) {
            console.warn('Emotional trends chart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Prepare data from sentiment trends
        const sentimentTrends = this.analyticsData.sentiment_trends || [];
        const last7Days = this.getLast7Days();
        
        // Group sentiment data by day
        const dailySentiments = {};
        sentimentTrends.forEach(trend => {
            if (trend.timestamp) {
                const date = new Date(trend.timestamp).toISOString().split('T')[0];
                if (!dailySentiments[date]) {
                    dailySentiments[date] = [];
                }
                dailySentiments[date].push(trend.sentiment || 0.5);
            }
        });
        
        // Calculate daily averages
        const sentimentData = last7Days.map(date => {
            const dayData = dailySentiments[date] || [];
            return dayData.length > 0 
                ? dayData.reduce((a, b) => a + b, 0) / dayData.length 
                : 0.5;
        });
        
        // Prepare emotional states data
        const emotionalStates = this.analyticsData.usage_stats?.emotional_states || {};
        const emotionNames = Object.keys(emotionalStates);
        
        const datasets = [
            {
                label: 'Average Sentiment',
                data: sentimentData,
                borderColor: this.colorPalette.primary,
                backgroundColor: this.colorPalette.primary + '20',
                fill: false,
                tension: 0.4
            }
        ];
        
        // Add emotion trend lines for significant emotions
        const significantEmotions = emotionNames
            .filter(emotion => emotionalStates[emotion] > 2)
            .slice(0, 3); // Top 3 emotions
        
        significantEmotions.forEach((emotion, index) => {
            // Simulate emotion trend data (in real implementation, this would come from historical data)
            const emotionData = last7Days.map(() => Math.random() * 0.3 + 0.2);
            
            datasets.push({
                label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                data: emotionData,
                borderColor: this.colorPalette.emotions[emotion] || this.colorPalette.secondary,
                backgroundColor: (this.colorPalette.emotions[emotion] || this.colorPalette.secondary) + '15',
                fill: false,
                tension: 0.4,
                borderWidth: 1
            });
        });
        
        this.charts.emotionalTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: last7Days.map(date => this.formatDateLabel(date)),
                datasets: datasets
            },
            options: this.chartConfigs.emotionalTrends.options
        });
    }
    
    initializeInteractionTypesChart() {
        const canvas = document.getElementById('interactionTypesChart');
        if (!canvas) {
            console.warn('Interaction types chart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        const interactionTypes = this.analyticsData.usage_stats?.interaction_types || {};
        const labels = Object.keys(interactionTypes);
        const data = Object.values(interactionTypes);
        
        if (labels.length === 0) {
            // Show empty state
            this.showChartEmptyState(canvas, 'No interaction data available yet');
            return;
        }
        
        // Generate colors for each interaction type
        const backgroundColors = labels.map(label => 
            this.colorPalette.interactions[label] || this.generateColor(label)
        );
        
        this.charts.interactionTypes = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.map(label => this.formatLabel(label)),
                datasets: [{
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => this.darkenColor(color)),
                    borderWidth: 2
                }]
            },
            options: this.chartConfigs.interactionTypes.options
        });
    }
    
    async updatePredictiveInsights() {
        const insightsContainer = document.getElementById('insightsContainer');
        if (!insightsContainer) return;
        
        try {
            const predictions = this.analyticsData.predictions || {};
            const modelStatus = predictions.model_status || {};
            const availableModels = predictions.available_models || [];
            
            let insightsHtml = '';
            
            if (availableModels.length === 0) {
                insightsHtml += `
                    <div class="alert alert-info" role="alert">
                        <i data-feather="info" class="me-2"></i>
                        <small>Building predictive models from conversation data...</small>
                    </div>
                `;
            } else {
                // Show model status
                insightsHtml += '<div class="row">';
                
                availableModels.slice(0, 3).forEach(modelName => {
                    const status = modelStatus[modelName] || {};
                    const accuracy = status.accuracy_score || 0;
                    const isActive = status.last_trained !== null;
                    
                    insightsHtml += `
                        <div class="col-12 mb-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <small class="text-capitalize">${modelName.replace('_', ' ')}</small>
                                <div class="d-flex align-items-center">
                                    ${isActive ? 
                                        `<span class="badge bg-success me-2">${(accuracy * 100).toFixed(0)}%</span>` :
                                        `<span class="badge bg-secondary me-2">Training</span>`
                                    }
                                    <i data-feather="${isActive ? 'check-circle' : 'clock'}" 
                                       class="text-${isActive ? 'success' : 'warning'}" 
                                       style="width: 14px; height: 14px;"></i>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                insightsHtml += '</div>';
                
                // Add insights based on patterns
                const insights = this.generateInsights();
                if (insights.length > 0) {
                    insightsHtml += '<div class="mt-3">';
                    insights.forEach(insight => {
                        insightsHtml += `
                            <div class="alert alert-${insight.type}" role="alert">
                                <i data-feather="${insight.icon}" class="me-2"></i>
                                <small>${insight.message}</small>
                            </div>
                        `;
                    });
                    insightsHtml += '</div>';
                }
            }
            
            insightsContainer.innerHTML = insightsHtml;
            feather.replace();
            
        } catch (error) {
            console.error('Predictive insights update failed:', error);
            insightsContainer.innerHTML = `
                <div class="alert alert-warning" role="alert">
                    <i data-feather="alert-triangle" class="me-2"></i>
                    <small>Unable to load predictive insights</small>
                </div>
            `;
            feather.replace();
        }
    }
    
    generateInsights() {
        const insights = [];
        const usageStats = this.analyticsData.usage_stats || {};
        const interactionTypes = usageStats.interaction_types || {};
        const emotionalStates = usageStats.emotional_states || {};
        
        // Analyze interaction patterns
        const totalInteractions = Object.values(interactionTypes).reduce((a, b) => a + b, 0);
        
        if (totalInteractions > 0) {
            // Check for dominant interaction types
            const sortedTypes = Object.entries(interactionTypes)
                .sort(([,a], [,b]) => b - a);
            
            if (sortedTypes.length > 0) {
                const [dominantType, count] = sortedTypes[0];
                const percentage = (count / totalInteractions * 100).toFixed(0);
                
                if (percentage > 50) {
                    if (dominantType === 'support_request') {
                        insights.push({
                            type: 'info',
                            icon: 'heart',
                            message: `${percentage}% of interactions are support requests. The AI is providing meaningful emotional assistance.`
                        });
                    } else if (dominantType === 'question') {
                        insights.push({
                            type: 'success',
                            icon: 'help-circle',
                            message: `${percentage}% of interactions are questions. Strong learning engagement detected.`
                        });
                    }
                }
            }
        }
        
        // Analyze emotional patterns
        const totalEmotions = Object.values(emotionalStates).reduce((a, b) => a + b, 0);
        
        if (totalEmotions > 0) {
            const positiveEmotions = (emotionalStates.happy || 0) + (emotionalStates.excited || 0);
            const negativeEmotions = (emotionalStates.sad || 0) + (emotionalStates.angry || 0) + (emotionalStates.worried || 0);
            
            const positiveRatio = positiveEmotions / totalEmotions;
            const negativeRatio = negativeEmotions / totalEmotions;
            
            if (positiveRatio > 0.6) {
                insights.push({
                    type: 'success',
                    icon: 'smile',
                    message: 'Predominantly positive emotional interactions detected. Relationship is healthy and supportive.'
                });
            } else if (negativeRatio > 0.6) {
                insights.push({
                    type: 'warning',
                    icon: 'alert-circle',
                    message: 'High frequency of challenging emotions. AI is providing important emotional support.'
                });
            }
        }
        
        // Check identity evolution
        const identityEvolution = this.analyticsData.identity_evolution || [];
        if (identityEvolution.length > 1) {
            const latestCoherence = identityEvolution[identityEvolution.length - 1]?.coherence || 0;
            
            if (latestCoherence > 0.7) {
                insights.push({
                    type: 'info',
                    icon: 'layers',
                    message: 'AI identity coherence is high. Personality development is progressing well.'
                });
            }
        }
        
        return insights;
    }
    
    async refreshAnalytics() {
        try {
            await this.loadAnalyticsData();
            this.updateCharts();
            this.updatePredictiveInsights();
            
            this.showRefreshSuccess();
        } catch (error) {
            console.error('Analytics refresh failed:', error);
            this.showError('Failed to refresh analytics data');
        }
    }
    
    updateCharts() {
        // Update emotional trends chart
        if (this.charts.emotionalTrends) {
            this.charts.emotionalTrends.destroy();
            this.initializeEmotionalTrendsChart();
        }
        
        // Update interaction types chart
        if (this.charts.interactionTypes) {
            this.charts.interactionTypes.destroy();
            this.initializeInteractionTypesChart();
        }
    }
    
    startAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            this.refreshAnalytics();
        }, this.refreshRate);
    }
    
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    // Utility methods
    getLast7Days() {
        const days = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            days.push(date.toISOString().split('T')[0]);
        }
        return days;
    }
    
    formatDateLabel(dateString) {
        const date = new Date(dateString);
        const today = new Date().toISOString().split('T')[0];
        const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
        
        if (dateString === today) return 'Today';
        if (dateString === yesterday) return 'Yesterday';
        
        return date.toLocaleDateString('en-US', { weekday: 'short' });
    }
    
    formatLabel(label) {
        return label
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    setupTabListeners() {
        // Re-initialize Feather icons when tabs are shown
        const tabButtons = document.querySelectorAll('#analyticsTabs button[data-bs-toggle="tab"]');
        tabButtons.forEach(button => {
            button.addEventListener('shown.bs.tab', (event) => {
                // Re-render Feather icons after tab switch
                feather.replace();
                
                // Re-initialize charts if needed when tab becomes visible
                const targetId = event.target.getAttribute('data-bs-target');
                if (targetId === '#emotional' && !this.charts.emotionalTrends) {
                    this.initializeEmotionalTrendsChart();
                } else if (targetId === '#interaction' && !this.charts.interactionTypes) {
                    this.initializeInteractionTypesChart();
                }
            });
        });
    }
    
    generateColor(seed) {
        // Generate a consistent color based on string seed
        let hash = 0;
        for (let i = 0; i < seed.length; i++) {
            const char = seed.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        
        const hue = Math.abs(hash) % 360;
        return `hsl(${hue}, 70%, 60%)`;
    }
    
    darkenColor(color) {
        // Simple color darkening for borders
        if (color.startsWith('#')) {
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            
            return `rgb(${Math.max(0, r - 30)}, ${Math.max(0, g - 30)}, ${Math.max(0, b - 30)})`;
        }
        return color;
    }
    
    showChartEmptyState(canvas, message) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#6c757d';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }
    
    showError(message) {
        console.error(message);
        
        // Create or update error notification
        let errorDiv = document.getElementById('analyticsError');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'analyticsError';
            errorDiv.className = 'alert alert-danger mt-3';
            
            const analyticsContainer = document.querySelector('.card:has(#emotionalTrendsChart)');
            if (analyticsContainer) {
                analyticsContainer.appendChild(errorDiv);
            }
        }
        
        errorDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <i data-feather="alert-triangle" class="me-2"></i>
                <span>${message}</span>
                <button type="button" class="btn btn-sm btn-outline-light ms-auto" onclick="analytics.refreshAnalytics()">
                    <i data-feather="refresh-cw"></i> Retry
                </button>
            </div>
        `;
        
        feather.replace();
    }
    
    showRefreshSuccess() {
        // Brief success indication
        const refreshButton = document.querySelector('button[onclick="refreshAnalytics()"]');
        if (refreshButton) {
            const originalHtml = refreshButton.innerHTML;
            refreshButton.innerHTML = '<i data-feather="check"></i>';
            refreshButton.classList.add('btn-success');
            
            feather.replace();
            
            setTimeout(() => {
                refreshButton.innerHTML = originalHtml;
                refreshButton.classList.remove('btn-success');
                feather.replace();
            }, 2000);
        }
    }
    
    destroy() {
        this.stopAutoRefresh();
        
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        this.charts = {};
    }
}

// Global analytics instance
let analytics;

// Initialize analytics
function initializeAnalytics() {
    analytics = new MotherXAnalytics();
    analytics.initialize().catch(error => {
        console.error('Failed to initialize analytics:', error);
    });
}

// Global refresh function for the refresh button
function refreshAnalytics() {
    if (analytics) {
        analytics.refreshAnalytics();
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (analytics) {
        analytics.destroy();
    }
});

// Handle visibility change to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (analytics) {
        if (document.hidden) {
            analytics.stopAutoRefresh();
        } else {
            analytics.startAutoRefresh();
        }
    }
});

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MotherXAnalytics;
}
