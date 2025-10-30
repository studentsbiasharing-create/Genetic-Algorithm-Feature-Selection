// ============================================
// Global Variables
// ============================================
let uploadedFile = null;
let currentResults = null;

// ============================================
// Initialize Upload Functionality
// ============================================
function initializeUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    // Load existing files
    loadExistingFiles();
    
    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--primary-color)';
        uploadZone.style.background = 'rgba(99, 102, 241, 0.1)';
    });
    
    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--border-color)';
        uploadZone.style.background = 'var(--card-bg)';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'var(--border-color)';
        uploadZone.style.background = 'var(--card-bg)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// ============================================
// Load Existing Files
// ============================================
function loadExistingFiles() {
    const filesList = document.getElementById('filesList');
    
    fetch('/datasets')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.files.length > 0) {
                displayExistingFiles(data.files);
            } else {
                filesList.innerHTML = '<p class="no-files"><i class="fas fa-inbox"></i> No datasets available</p>';
            }
        })
        .catch(error => {
            filesList.innerHTML = '<p class="error-files"><i class="fas fa-exclamation-triangle"></i> Error loading datasets</p>';
            console.error('Error loading files:', error);
        });
}

// ============================================
// Display Existing Files
// ============================================
function displayExistingFiles(files) {
    const filesList = document.getElementById('filesList');
    
    const filesHTML = files.map(file => {
        const date = new Date(file.modified * 1000).toLocaleDateString();
        const size = formatFileSize(file.size);
        
        return `
            <div class="file-item" onclick="selectExistingFile('${file.filename}')">
                <div class="file-icon">
                    <i class="fas fa-file-csv"></i>
                </div>
                <div class="file-details">
                    <h4>${file.filename}</h4>
                    <p><i class="fas fa-weight"></i> ${size} &nbsp;•&nbsp; <i class="fas fa-calendar"></i> ${date}</p>
                </div>
                <div class="file-action">
                    <i class="fas fa-chevron-right"></i>
                </div>
            </div>
        `;
    }).join('');
    
    filesList.innerHTML = filesHTML;
}

// ============================================
// Format File Size
// ============================================
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================
// Select Existing File
// ============================================
function selectExistingFile(filename) {
    showLoading('Loading dataset...');
    
    // Get the upload folder path
    const filepath = 'uploads/' + filename;
    
    // Create FormData and upload the existing file reference
    fetch('/upload', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            existing_file: filename
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            uploadedFile = data;
            displayFileInfo(data);
            displayDatasetInfo(data.dataset_info);
            showConfigSection();
            showNotification(`Dataset "${filename}" loaded successfully!`, 'success');
        } else {
            showNotification(data.error || 'Failed to load dataset', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showNotification('Error loading dataset: ' + error.message, 'error');
    });
}

// ============================================
// Handle File Selection
// ============================================
function handleFileSelect(file) {
    if (!file.name.endsWith('.csv')) {
        showNotification('Please select a CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading('Uploading and validating dataset...');
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            uploadedFile = data;
            displayFileInfo(data);
            displayDatasetInfo(data.dataset_info);
            showConfigSection();
            showNotification('Dataset uploaded successfully!', 'success');
        } else {
            showNotification(data.error || 'Upload failed', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showNotification('Error uploading file: ' + error.message, 'error');
    });
}

// ============================================
// Display File Info
// ============================================
function displayFileInfo(data) {
    document.getElementById('fileName').textContent = data.filename;
    const feats = (data.dataset_info && typeof data.dataset_info.n_features === 'number') 
        ? data.dataset_info.n_features 
        : Math.max(0, (data.validation && typeof data.validation.n_features === 'number') ? data.validation.n_features - 1 : 0);
    document.getElementById('fileStats').textContent = 
        `${data.validation.n_samples} samples, ${feats} features`;
    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('uploadZone').style.display = 'none';
}

// ============================================
// Display Dataset Info
// ============================================
function displayDatasetInfo(info) {
    document.getElementById('nSamples').textContent = info.n_samples.toLocaleString();
    document.getElementById('nFeatures').textContent = info.n_features.toLocaleString();
    document.getElementById('nClasses').textContent = info.n_classes.toLocaleString();
    document.getElementById('missingValues').textContent = info.missing_values.toLocaleString();
    document.getElementById('datasetInfo').classList.remove('hidden');
}

// ============================================
// Remove File
// ============================================
function removeFile() {
    uploadedFile = null;
    document.getElementById('fileInfo').classList.add('hidden');
    document.getElementById('datasetInfo').classList.add('hidden');
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('configSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('fileInput').value = '';
}

// ============================================
// Show Configuration Section
// ============================================
function showConfigSection() {
    document.getElementById('configSection').classList.remove('hidden');
    
    // Smooth scroll
    setTimeout(() => {
        document.getElementById('configSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }, 300);
}

// ============================================
// Initialize Parameter Sliders
// ============================================
function initializeParamSliders() {
    const sliders = [
        { id: 'populationSize', valueId: 'populationSizeValue', format: v => v },
        { id: 'generations', valueId: 'generationsValue', format: v => v },
        { id: 'crossoverRate', valueId: 'crossoverRateValue', format: v => v.toFixed(2) },
        { id: 'mutationRate', valueId: 'mutationRateValue', format: v => v.toFixed(2) },
        { id: 'eliteSize', valueId: 'eliteSizeValue', format: v => v }
    ];
    
    sliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);
        
        if (element && valueElement) {
            element.addEventListener('input', (e) => {
                valueElement.textContent = slider.format(parseFloat(e.target.value));
            });
        }
    });
}

// ============================================
// Reset Configuration
// ============================================
function resetConfig() {
    document.getElementById('populationSize').value = 50;
    document.getElementById('populationSizeValue').textContent = '50';
    document.getElementById('generations').value = 100;
    document.getElementById('generationsValue').textContent = '100';
    document.getElementById('crossoverRate').value = 0.8;
    document.getElementById('crossoverRateValue').textContent = '0.80';
    document.getElementById('mutationRate').value = 0.1;
    document.getElementById('mutationRateValue').textContent = '0.10';
    document.getElementById('eliteSize').value = 5;
    document.getElementById('eliteSizeValue').textContent = '5';
    
    showNotification('Parameters reset to defaults', 'info');
}

// ============================================
// Run Analysis
// ============================================
function runAnalysis() {
    if (!uploadedFile) {
        showNotification('Please upload a dataset first', 'error');
        return;
    }
    
    const params = {
        filepath: uploadedFile.filepath,
        target_column: uploadedFile.dataset_info.target_name,
        population_size: parseInt(document.getElementById('populationSize').value),
        generations: parseInt(document.getElementById('generations').value),
        crossover_rate: parseFloat(document.getElementById('crossoverRate').value),
        mutation_rate: parseFloat(document.getElementById('mutationRate').value),
        elite_size: parseInt(document.getElementById('eliteSize').value),
        n_jobs: -1  // Use all CPU cores
    };
    
    showLoading('Running feature selection analysis...<br><small>Using all CPU cores for faster processing</small>');
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            currentResults = data;
            displayResults(data);
            showNotification('Analysis completed successfully!', 'success');
        } else {
            showNotification(data.error || 'Analysis failed', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showNotification('Error running analysis: ' + error.message, 'error');
    });
}

// ============================================
// Display Results
// ============================================
function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Create summary cards
    const summaryHTML = `
        <div class="summary-card">
            <i class="fas fa-dna"></i>
            <span class="value">${data.ga_results.n_selected}</span>
            <span class="label">Selected Features</span>
        </div>
        <div class="summary-card">
            <i class="fas fa-chart-line"></i>
            <span class="value">${(data.ga_results.best_fitness * 100).toFixed(2)}%</span>
            <span class="label">Best Fitness</span>
        </div>
        <div class="summary-card">
            <i class="fas fa-percentage"></i>
            <span class="value">${((data.ga_results.n_selected / data.ga_results.n_total) * 100).toFixed(1)}%</span>
            <span class="label">Feature Reduction</span>
        </div>
        <div class="summary-card">
            <i class="fas fa-clock"></i>
            <span class="value">${data.ga_results.execution_time.toFixed(2)}s</span>
            <span class="label">Execution Time</span>
        </div>
    `;
    
    document.getElementById('resultsSummary').innerHTML = summaryHTML;
    
    // Display overview tab by default
    switchTab('overview');
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }, 300);
}

// ============================================
// Switch Tabs
// ============================================
function switchTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    let activeBtn = null;
    if (typeof event !== 'undefined' && event && event.target) {
        activeBtn = event.target.closest('.tab-btn');
    }
    if (!activeBtn) {
        activeBtn = document.querySelector(`.tab-btn[onclick="switchTab('${tabName}')"]`) || document.querySelector('.tab-btn');
    }
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Display tab content
    const tabContent = document.getElementById('tabContent');
    
    switch(tabName) {
        case 'overview':
            displayOverviewTab(tabContent);
            break;
        case 'ga':
            displayGATab(tabContent);
            break;
        case 'statistical':
            displayStatisticalTab(tabContent);
            break;
        case 'comparison':
            displayComparisonTab(tabContent);
            break;
    }
}

// ============================================
// Display Overview Tab
// ============================================
function displayOverviewTab(container) {
    if (!currentResults) return;
    
    const html = `
        <h3 class="mb-3">Analysis Overview</h3>
        
        <div class="chart-container">
            <h4>Convergence Plot</h4>
            <img src="${currentResults.visualizations.convergence}" alt="Convergence Plot">
        </div>
        
        <div class="chart-container">
            <h4>Population Diversity</h4>
            <img src="${currentResults.visualizations.diversity}" alt="Diversity Plot">
        </div>
        
        <div class="chart-container">
            <h4>Method Comparison</h4>
            <img src="${currentResults.visualizations.comparison}" alt="Comparison Plot">
        </div>
    `;
    
    container.innerHTML = html;
}

// ============================================
// Display GA Tab
// ============================================
function displayGATab(container) {
    if (!currentResults) return;
    
    const ga = currentResults.ga_results;
    const featureList = ga.selected_feature_names.map((name, idx) => 
        `<li><strong>${name}</strong> (Index: ${ga.selected_features[idx]})</li>`
    ).join('');
    
    const html = `
        <h3 class="mb-3">Genetic Algorithm Results</h3>
        
        <div class="info-grid mb-4">
            <div class="info-card">
                <i class="fas fa-check-circle"></i>
                <div>
                    <span class="label">Selected Features</span>
                    <span class="value">${ga.n_selected}</span>
                </div>
            </div>
            <div class="info-card">
                <i class="fas fa-trophy"></i>
                <div>
                    <span class="label">Best Fitness</span>
                    <span class="value">${(ga.best_fitness * 100).toFixed(2)}%</span>
                </div>
            </div>
            <div class="info-card">
                <i class="fas fa-users"></i>
                <div>
                    <span class="label">Population Size</span>
                    <span class="value">${ga.parameters.population_size}</span>
                </div>
            </div>
            <div class="info-card">
                <i class="fas fa-sync"></i>
                <div>
                    <span class="label">Generations</span>
                    <span class="value">${ga.parameters.generations}</span>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h4>Selected Features</h4>
            <img src="${currentResults.visualizations.feature_importance.ga}" alt="GA Feature Importance">
        </div>
        
        <div class="mt-4">
            <h4>Selected Feature Names:</h4>
            <ul style="columns: 2; list-style-position: inside; color: var(--text-secondary);">
                ${featureList}
            </ul>
        </div>
    `;
    
    container.innerHTML = html;
}

// ============================================
// Display Statistical Tab
// ============================================
function displayStatisticalTab(container) {
    if (!currentResults) return;
    
    const methods = currentResults.statistical_results;
    
    let methodsHTML = '';
    for (const [key, method] of Object.entries(methods)) {
        const plot = currentResults.visualizations.feature_importance[key];
        if (plot) {
            methodsHTML += `
                <div class="chart-container">
                    <h4>${method.method}</h4>
                    <p style="color: var(--text-muted);">Execution Time: ${method.execution_time.toFixed(4)}s</p>
                    <img src="${plot}" alt="${method.method}">
                    <div class="mt-2">
                        <strong>Top Features:</strong> ${method.top_feature_names.slice(0, 5).join(', ')}
                    </div>
                </div>
            `;
        }
    }
    
    const html = `
        <h3 class="mb-3">Statistical Methods Results</h3>
        ${methodsHTML}
    `;
    
    container.innerHTML = html;
}

// ============================================
// Display Comparison Tab
// ============================================
function displayComparisonTab(container) {
    if (!currentResults) return;
    
    const comparison = currentResults.comparison;
    
    let tableRows = '';
    for (const [key, result] of Object.entries(comparison)) {
        if (result.method) {
            tableRows += `
                <tr>
                    <td style="font-weight: 600;">${result.method || key.toUpperCase()}</td>
                    <td>${result.n_features}</td>
                    <td>${(result.accuracy * 100).toFixed(2)}%</td>
                    <td>${result.execution_time ? result.execution_time.toFixed(4) + 's' : 'N/A'}</td>
                    <td>${result.overlap_with_ga || 'N/A'}</td>
                </tr>
            `;
        }
    }
    
    const html = `
        <h3 class="mb-3">Method Comparison</h3>
        
        <div class="chart-container">
            <img src="${currentResults.visualizations.comparison}" alt="Comparison Chart">
        </div>
        
        <div class="chart-container">
            <h4>Feature Overlap Heatmap</h4>
            <img src="${currentResults.visualizations.overlap_heatmap}" alt="Overlap Heatmap">
        </div>
        
        <div class="mt-4">
            <h4>Detailed Comparison Table</h4>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                    <thead>
                        <tr style="background: var(--dark-bg); text-align: left;">
                            <th style="padding: 1rem; border-bottom: 2px solid var(--border-color);">Method</th>
                            <th style="padding: 1rem; border-bottom: 2px solid var(--border-color);">Features</th>
                            <th style="padding: 1rem; border-bottom: 2px solid var(--border-color);">Accuracy</th>
                            <th style="padding: 1rem; border-bottom: 2px solid var(--border-color);">Time</th>
                            <th style="padding: 1rem; border-bottom: 2px solid var(--border-color);">Overlap with GA</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ============================================
// Loading Overlay
// ============================================
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    const text = document.getElementById('loadingText');
    text.innerHTML = message;
    overlay.classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

// ============================================
// Notifications
// ============================================
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: var(--card-bg);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: var(--shadow-xl);
        border-left: 4px solid ${type === 'success' ? 'var(--success-color)' : type === 'error' ? 'var(--danger-color)' : 'var(--primary-color)'};
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
    `;
    
    const icon = type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle';
    notification.innerHTML = `
        <i class="fas fa-${icon}" style="margin-right: 0.5rem;"></i>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
