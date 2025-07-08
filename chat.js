// Enhanced chat functionality for MotherX AI
class MotherXChat {
    constructor() {
        this.chatForm = document.getElementById('chatForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        this.isTyping = false;
        this.messageHistory = [];
        this.lastInteractionTime = Date.now();
        
        this.initializeEventListeners();
        this.startHeartbeat();
    }
    
    initializeEventListeners() {
        // Form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Enter key handling
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Input focus handling
        this.messageInput.addEventListener('focus', () => {
            this.scrollToBottom();
        });
        
        // Auto-resize input
        this.messageInput.addEventListener('input', () => {
            this.adjustInputHeight();
        });
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message to UI
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.adjustInputHeight();
        
        // Show typing indicator
        this.showTyping();
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            this.hideTyping();
            
            if (data.error) {
                this.addMessage(`I apologize, but I encountered an error: ${data.error}`, 'ai', 'error');
            } else {
                // Add AI response with enhanced metadata
                this.addMessage(data.response, 'ai', 'normal', {
                    sentiment: data.sentiment,
                    intent: data.intent,
                    emotional_context: data.emotional_context,
                    predicted_state: data.predicted_state,
                    identity_evolution: data.identity_evolution,
                    autonomous_action: data.autonomous_action
                });
                
                // Update interface based on response data
                this.updateInterfaceFromResponse(data);
                
                // Handle autonomous actions
                if (data.autonomous_action) {
                    this.handleAutonomousAction(data.autonomous_action);
                }
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTyping();
            this.addMessage(
                "I'm having trouble connecting right now. Please check your connection and try again.", 
                'ai', 
                'error'
            );
        }
        
        this.lastInteractionTime = Date.now();
    }
    
    addMessage(text, sender, type = 'normal', metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message fade-in`;
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        let statusIndicator = '';
        if (type === 'error') {
            statusIndicator = '<i data-feather="alert-triangle" class="text-danger"></i>';
        } else if (metadata.autonomous_action) {
            statusIndicator = '<i data-feather="cpu" class="text-info"></i>';
        } else if (metadata.identity_evolution > 0.7) {
            statusIndicator = '<i data-feather="layers" class="text-warning"></i>';
        }
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <strong>${sender === 'ai' ? window.assistantName : window.userName}</strong>
                    <div class="d-flex align-items-center">
                        ${statusIndicator}
                        <small class="text-muted ms-2">${timestamp}</small>
                    </div>
                </div>
                <div class="message-text">
                    ${this.formatMessage(text)}
                </div>
                ${this.createMessageMetadata(metadata)}
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        
        // Initialize feather icons in the new message
        feather.replace();
        
        this.scrollToBottom();
        this.messageHistory.push({ text, sender, timestamp: Date.now(), metadata });
        
        // Limit message history
        if (this.messageHistory.length > 100) {
            this.messageHistory = this.messageHistory.slice(-100);
        }
    }
    
    formatMessage(text) {
        // Basic markdown-like formatting
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        text = text.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Convert newlines to paragraphs
        const paragraphs = text.split('\n\n');
        return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
    }
    
    createMessageMetadata(metadata) {
        if (!metadata || Object.keys(metadata).length === 0) return '';
        
        let metadataHtml = '<div class="message-metadata mt-2 pt-2 border-top">';
        
        if (metadata.sentiment !== undefined) {
            const sentimentColor = metadata.sentiment > 0.6 ? 'success' : 
                                 metadata.sentiment < 0.4 ? 'danger' : 'warning';
            metadataHtml += `
                <small class="d-block text-${sentimentColor}">
                    <i data-feather="heart"></i> Sentiment: ${(metadata.sentiment * 100).toFixed(0)}%
                </small>
            `;
        }
        
        if (metadata.intent) {
            metadataHtml += `
                <small class="d-block text-info">
                    <i data-feather="target"></i> Intent: ${metadata.intent.replace('_', ' ')}
                </small>
            `;
        }
        
        if (metadata.emotional_context && metadata.emotional_context.dominant_emotion !== 'neutral') {
            metadataHtml += `
                <small class="d-block text-warning">
                    <i data-feather="zap"></i> Emotion: ${metadata.emotional_context.dominant_emotion}
                </small>
            `;
        }
        
        if (metadata.identity_evolution > 0.5) {
            metadataHtml += `
                <small class="d-block text-primary">
                    <i data-feather="trending-up"></i> Identity Evolution: ${(metadata.identity_evolution * 100).toFixed(0)}%
                </small>
            `;
        }
        
        metadataHtml += '</div>';
        
        return metadataHtml;
    }
    
    updateInterfaceFromResponse(data) {
        // Update identity coherence if provided
        if (data.identity_evolution !== undefined) {
            this.updateIdentityCoherence(data.identity_evolution);
        }
        
        // Update mood indicator based on emotional context
        if (data.emotional_context) {
            this.updateMoodIndicator(data.emotional_context);
        }
        
        // Trigger panel updates
        setTimeout(() => {
            updateMemoryPanel();
            updateIdentityPanel();
        }, 1000);
    }
    
    updateIdentityCoherence(coherence) {
        const progressBar = document.querySelector('.progress-bar.bg-info');
        if (progressBar) {
            progressBar.style.width = `${(coherence * 100).toFixed(0)}%`;
            
            const coherenceText = progressBar.parentElement.nextElementSibling;
            if (coherenceText) {
                coherenceText.textContent = `${(coherence * 100).toFixed(0)}%`;
            }
        }
    }
    
    updateMoodIndicator(emotionalContext) {
        const moodIndicator = document.getElementById('moodIndicator');
        if (!moodIndicator) return;
        
        const emotion = emotionalContext.dominant_emotion || 'neutral';
        const intensity = emotionalContext.emotional_intensity || 0.5;
        
        // Update icon and color based on emotion
        const iconMap = {
            'happy': { icon: 'smile', color: 'text-success' },
            'sad': { icon: 'frown', color: 'text-info' },
            'angry': { icon: 'zap', color: 'text-danger' },
            'excited': { icon: 'star', color: 'text-warning' },
            'worried': { icon: 'cloud', color: 'text-secondary' },
            'neutral': { icon: 'heart', color: 'text-warning' }
        };
        
        const iconData = iconMap[emotion] || iconMap['neutral'];
        
        moodIndicator.innerHTML = `<i data-feather="${iconData.icon}" class="${iconData.color}"></i>`;
        
        // Add pulsing effect for high intensity
        if (intensity > 0.7) {
            moodIndicator.classList.add('pulse-animation');
        } else {
            moodIndicator.classList.remove('pulse-animation');
        }
        
        feather.replace();
    }
    
    handleAutonomousAction(action) {
        // Show notification for autonomous actions
        if (action && action.type) {
            this.showNotification(
                `Autonomous Action: ${action.type.replace('_', ' ')}`,
                'info'
            );
            
            // Update autonomous panel
            setTimeout(updateAutonomousPanel, 500);
        }
    }
    
    showTyping() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'flex';
        this.sendButton.disabled = true;
        this.scrollToBottom();
    }
    
    hideTyping() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
        this.sendButton.disabled = false;
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    adjustInputHeight() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show notification`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            animation: slideInRight 0.3s ease-out;
        `;
        
        notification.innerHTML = `
            <div class="d-flex align-items-center">
                <i data-feather="bell" class="me-2"></i>
                <span>${message}</span>
                <button type="button" class="btn-close ms-auto" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        document.body.appendChild(notification);
        feather.replace();
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
    
    startHeartbeat() {
        // Check for autonomous actions every 30 seconds
        setInterval(() => {
            const timeSinceLastInteraction = Date.now() - this.lastInteractionTime;
            
            // If no interaction for 5 minutes, check for autonomous actions
            if (timeSinceLastInteraction > 300000) {
                this.checkForAutonomousActions();
            }
        }, 30000);
    }
    
    async checkForAutonomousActions() {
        try {
            const response = await fetch('/autonomous');
            if (response.ok) {
                const data = await response.json();
                
                if (data.pending_actions && data.pending_actions.length > 0) {
                    updateAutonomousPanel();
                }
            }
        } catch (error) {
            console.error('Autonomous check error:', error);
        }
    }
}

// Panel update functions
async function updateMemoryPanel() {
    try {
        const response = await fetch('/memory');
        if (response.ok) {
            const data = await response.json();
            
            // Update user facts
            document.getElementById('userName').textContent = data.name || 'Guest';
            document.getElementById('userLocation').textContent = data.location || 'Unknown';
            
            // Update interaction count
            const interactionCountEl = document.querySelector('.h6.text-primary');
            if (interactionCountEl) {
                interactionCountEl.textContent = data.interaction_count || 0;
            }
        }
    } catch (error) {
        console.error('Memory panel update error:', error);
    }
}

async function updateIdentityPanel() {
    try {
        const response = await fetch('/identity');
        if (response.ok) {
            const data = await response.json();
            
            // Update personality traits
            updatePersonalityTraits(data.identity_state.personality_traits);
            
            // Update core values
            updateCoreValues(data.identity_state.core_values);
            
            // Update mood state
            updateMoodState(data.identity_state.mood_state);
            
            // Update coherence score
            if (data.coherence_score !== undefined) {
                chat.updateIdentityCoherence(data.coherence_score);
            }
        }
    } catch (error) {
        console.error('Identity panel update error:', error);
    }
}

async function updateAutonomousPanel() {
    try {
        const response = await fetch('/autonomous');
        if (response.ok) {
            const data = await response.json();
            
            const autonomousStatus = document.getElementById('autonomousStatus');
            if (autonomousStatus && data.recent_decisions) {
                updateAutonomousActions(data.recent_decisions);
            }
        }
    } catch (error) {
        console.error('Autonomous panel update error:', error);
    }
}

function updatePersonalityTraits(traits) {
    const traitsContainer = document.getElementById('personalityTraits');
    if (!traitsContainer || !traits) return;
    
    let traitsHtml = '';
    for (const [trait, value] of Object.entries(traits)) {
        if (value > 0.6) {
            traitsHtml += `
                <div class="d-flex justify-content-between mb-2">
                    <span class="text-capitalize">${trait.replace('_', ' ')}</span>
                    <div class="progress flex-grow-1 ms-2 me-2" style="height: 6px;">
                        <div class="progress-bar bg-primary" style="width: ${(value * 100).toFixed(0)}%"></div>
                    </div>
                    <small class="text-muted">${(value * 100).toFixed(0)}%</small>
                </div>
            `;
        }
    }
    
    traitsContainer.innerHTML = traitsHtml;
}

function updateCoreValues(values) {
    const valuesContainer = document.getElementById('coreValues');
    if (!valuesContainer || !values) return;
    
    const valuesHtml = values.slice(0, 4).map(value => 
        `<span class="badge bg-info me-1 mb-1">${value.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>`
    ).join('');
    
    valuesContainer.innerHTML = valuesHtml;
}

function updateMoodState(moodState) {
    const moodContainer = document.getElementById('moodState');
    if (!moodContainer || !moodState) return;
    
    let moodHtml = '';
    for (const [dimension, level] of Object.entries(moodState)) {
        const colorClass = level > 0.7 ? 'bg-success' : level > 0.4 ? 'bg-warning' : 'bg-secondary';
        
        moodHtml += `
            <div class="col-6 mb-2">
                <small class="text-muted text-capitalize">${dimension}</small>
                <div class="progress mt-1" style="height: 4px;">
                    <div class="progress-bar ${colorClass}" style="width: ${(level * 100).toFixed(0)}%"></div>
                </div>
            </div>
        `;
    }
    
    moodContainer.innerHTML = moodHtml;
}

function updateAutonomousActions(actions) {
    const autonomousStatus = document.getElementById('autonomousStatus');
    if (!autonomousStatus || !actions || actions.length === 0) return;
    
    let actionsHtml = '<h6 class="text-muted">Recent Autonomous Actions</h6><div class="autonomous-actions">';
    
    actions.slice(0, 3).forEach(action => {
        const iconClass = action.success ? 'check-circle text-success' : 'alert-circle text-warning';
        
        actionsHtml += `
            <div class="mb-2 p-2 border rounded">
                <div class="d-flex justify-content-between">
                    <small class="text-capitalize">${action.action_type.replace('_', ' ')}</small>
                    <i data-feather="${iconClass.split(' ')[0]}" class="${iconClass.split(' ')[1]}" style="width: 16px; height: 16px;"></i>
                </div>
                <small class="text-muted">${action.executed_at.substring(0, 16)}</small>
            </div>
        `;
    });
    
    actionsHtml += '</div>';
    autonomousStatus.innerHTML = actionsHtml;
    
    feather.replace();
}

// CSS for pulse animation
const style = document.createElement('style');
style.textContent = `
    .pulse-animation {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .notification {
        animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// Initialize chat
let chat;

function initializeChat() {
    chat = new MotherXChat();
    console.log('MotherX Chat initialized');
}
