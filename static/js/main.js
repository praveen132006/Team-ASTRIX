document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const mediaForm = document.getElementById('mediaForm');
    const textForm = document.getElementById('textForm');
    const mediaResults = document.getElementById('mediaResults');
    const textResults = document.getElementById('textResults');
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('mediaFile');
    const textInput = document.getElementById('textInput');
    const analyzeMediaBtn = document.getElementById('analyzeMediaBtn');
    const analyzeTextBtn = document.getElementById('analyzeTextBtn');
    const charCount = document.getElementById('charCount');
    const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
    const spinnerOverlay = document.querySelector('.spinner-overlay');
    const filePreview = document.querySelector('.file-preview');
    const filePreviewImg = document.getElementById('filePreviewImg');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFileBtn = document.getElementById('removeFile');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Tab switching functionality
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            
            // Remove active class from all tabs
            tabBtns.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Character counter for text analysis
    textInput.addEventListener('input', function() {
        const count = this.value.length;
        charCount.textContent = count;
        
        // Enable/disable analyze button based on character count
        if (count >= 100) {
            analyzeTextBtn.removeAttribute('disabled');
        } else {
            analyzeTextBtn.setAttribute('disabled', true);
        }
    });

    // File drag and drop handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        });
    });

    uploadArea.addEventListener('drop', handleDrop);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    }

    // File selection handling
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Show file preview
        const fileUrl = URL.createObjectURL(file);
        filePreviewImg.src = fileUrl;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        // Show preview, hide upload content
        document.querySelector('.upload-content').style.display = 'none';
        filePreview.style.display = 'block';
        
        // Enable analyze button
        analyzeMediaBtn.removeAttribute('disabled');
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // Remove file button
    removeFileBtn.addEventListener('click', function() {
        fileInput.value = '';
        filePreview.style.display = 'none';
        document.querySelector('.upload-content').style.display = 'block';
        analyzeMediaBtn.setAttribute('disabled', true);
    });

    // Handle media file analysis
    mediaForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!fileInput.files[0]) {
            showAlert(mediaResults, 'Please select a file to analyze', 'danger');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            // Show loading spinner
            showSpinner();
            
            const response = await fetch('/analyze/image', {
                method: 'POST',
                body: formData
            });
            
            // Hide spinner
            hideSpinner();
            
            const data = await response.json();
            
            if (response.ok) {
                displayMediaResults(data);
            } else {
                showAlert(mediaResults, data.error || 'An error occurred during analysis', 'danger');
            }
        } catch (error) {
            hideSpinner();
            showAlert(mediaResults, 'An error occurred during analysis', 'danger');
        }
    });

    // Handle text analysis
    textForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const text = textInput.value.trim();
        
        if (text.length < 100) {
            showAlert(textResults, 'Please enter at least 100 characters to analyze', 'danger');
            return;
        }
        
        try {
            // Show loading spinner
            showSpinner();
            
            const response = await fetch('/analyze/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });
            
            // Hide spinner
            hideSpinner();
            
            const data = await response.json();
            
            if (response.ok) {
                displayTextResults(data);
            } else {
                showAlert(textResults, data.error || 'An error occurred during analysis', 'danger');
            }
        } catch (error) {
            hideSpinner();
            showAlert(textResults, 'An error occurred during analysis', 'danger');
        }
    });

    function showSpinner() {
        spinnerOverlay.style.display = 'flex';
    }

    function hideSpinner() {
        spinnerOverlay.style.display = 'none';
    }

    function showAlert(container, message, type) {
        container.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>`;
    }

    function displayMediaResults(data) {
        const { deepfake_analysis, metadata_analysis } = data;
        const probability = deepfake_analysis.manipulation_probability * 100;
        const confidenceScore = deepfake_analysis.confidence_score * 100;
        
        let scoreClass, scoreLabel;
        
        if (probability > 75) {
            scoreClass = 'high';
            scoreLabel = 'High risk';
        } else if (probability > 40) {
            scoreClass = 'medium';
            scoreLabel = 'Medium risk';
        } else {
            scoreClass = 'low';
            scoreLabel = 'Low risk';
        }
        
        // Forensic markers
        const markers = deepfake_analysis.forensic_markers || [];
        const markerItems = markers.length > 0 
            ? markers.map(marker => `<li>${marker}</li>`).join('')
            : '<li>No suspicious markers detected</li>';
        
        // Visualizations
        const visualizations = deepfake_analysis.visualizations || {};
        const hasVisualizations = Object.keys(visualizations).some(key => visualizations[key]);
        
        // Classification result
        const classification = deepfake_analysis.classification || (probability > 50 ? 'AI-generated' : 'Authentic');
        const classificationClass = probability > 50 ? 'high' : 'low';
        
        const resultHtml = `
            <div class="result-card">
                <div class="result-header">
                    <h4>Analysis Results</h4>
                    <div class="result-score ${scoreClass}">${scoreLabel}: ${probability.toFixed(1)}%</div>
                </div>
                
                <div class="classification-badge mb-4">
                    <span class="badge bg-${classificationClass} fs-5 px-4 py-2">
                        <i class="${probability > 50 ? 'fas fa-robot' : 'fas fa-camera'} me-2"></i>
                        ${classification}
                    </span>
                </div>
                
                <div class="progress-container">
                    <div class="progress-label">
                        <span>Manipulation Probability</span>
                        <span class="text-${scoreClass}">${probability.toFixed(1)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-${scoreClass}" role="progressbar" 
                             style="width: ${probability}%" 
                             aria-valuenow="${probability}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>

                <div class="progress-container">
                    <div class="progress-label">
                        <span>Analysis Confidence</span>
                        <span>${confidenceScore.toFixed(1)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-info" role="progressbar" 
                             style="width: ${confidenceScore}%" 
                             aria-valuenow="${confidenceScore}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>
                
                ${hasVisualizations ? `
                <div class="forensic-visualizations mt-4">
                    <h5 class="mb-3">Forensic Visualizations</h5>
                    <div class="row">
                        ${visualizations.ela_heatmap ? `
                        <div class="col-md-6 mb-3">
                            <div class="viz-card">
                                <div class="viz-header">ELA Analysis</div>
                                <div class="viz-image">
                                    <img src="${visualizations.ela_heatmap}" alt="ELA Analysis" class="img-fluid">
                                </div>
                                <div class="viz-caption">Error Level Analysis highlighting potentially edited regions</div>
                            </div>
                        </div>` : ''}
                        
                        ${visualizations.noise_map ? `
                        <div class="col-md-6 mb-3">
                            <div class="viz-card">
                                <div class="viz-header">Noise Analysis</div>
                                <div class="viz-image">
                                    <img src="${visualizations.noise_map}" alt="Noise Analysis" class="img-fluid">
                                </div>
                                <div class="viz-caption">Noise pattern analysis revealing potential AI generation signs</div>
                            </div>
                        </div>` : ''}
                        
                        ${visualizations.cnn_heatmap ? `
                        <div class="col-md-6 mb-3">
                            <div class="viz-card">
                                <div class="viz-header">CNN Detection</div>
                                <div class="viz-image">
                                    <img src="${visualizations.cnn_heatmap}" alt="CNN Heatmap" class="img-fluid">
                                </div>
                                <div class="viz-caption">CNN classifier highlighting features suggesting AI generation</div>
                            </div>
                        </div>` : ''}
                        
                        ${visualizations.face_detection ? `
                        <div class="col-md-6 mb-3">
                            <div class="viz-card">
                                <div class="viz-header">Face Detection</div>
                                <div class="viz-image">
                                    <img src="${visualizations.face_detection}" alt="Face Detection" class="img-fluid">
                                </div>
                                <div class="viz-caption">Detected faces for facial manipulation analysis</div>
                            </div>
                        </div>` : ''}
                    </div>
                </div>` : ''}
                
                <div class="forensic-markers mt-4">
                    <h5>Forensic Markers Detected</h5>
                    <ul>
                        ${markerItems}
                    </ul>
                </div>

                <div class="result-message mt-4">
                    <i class="fas fa-info-circle me-2"></i> ${deepfake_analysis.recommendations}
                </div>
                
                <div class="result-actions">
                    <button class="btn btn-outline-primary" onclick="showDetailedAnalysis(${JSON.stringify(data).replace(/"/g, '&quot;')})">
                        <i class="fas fa-chart-line me-2"></i>Detailed Analysis
                    </button>
                    <button class="btn btn-outline-secondary" id="downloadReport">
                        <i class="fas fa-download me-2"></i>Download Report
                    </button>
                </div>
            </div>`;
        
        mediaResults.innerHTML = resultHtml;
        
        // Add CSS for visualization cards if not already added
        if (!document.getElementById('viz-styles')) {
            const styleEl = document.createElement('style');
            styleEl.id = 'viz-styles';
            styleEl.textContent = `
                .viz-card {
                    border-radius: var(--radius-md);
                    box-shadow: var(--shadow-sm);
                    overflow: hidden;
                    height: 100%;
                    border: 1px solid var(--gray-200);
                    background-color: var(--white);
                }
                .viz-header {
                    padding: 10px 15px;
                    background-color: var(--gray-100);
                    font-weight: 600;
                    font-size: 0.875rem;
                    border-bottom: 1px solid var(--gray-200);
                }
                .viz-image {
                    padding: 10px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .viz-image img {
                    max-height: 200px;
                    object-fit: contain;
                    width: 100%;
                }
                .viz-caption {
                    padding: 8px 15px;
                    font-size: 0.75rem;
                    color: var(--gray-600);
                    text-align: center;
                    border-top: 1px solid var(--gray-200);
                }
                .classification-badge {
                    text-align: center;
                }
            `;
            document.head.appendChild(styleEl);
        }
    }

    function displayTextResults(data) {
        const probability = data.ai_generated_probability * 100;
        
        let scoreClass, scoreLabel;
        
        if (probability > 75) {
            scoreClass = 'high';
            scoreLabel = 'High probability';
        } else if (probability > 40) {
            scoreClass = 'medium';
            scoreLabel = 'Medium probability';
        } else {
            scoreClass = 'low';
            scoreLabel = 'Low probability';
        }
        
        // Get recommendations based on score
        let recommendation;
        if (probability > 75) {
            recommendation = 'This text shows strong indicators of AI generation. Review carefully before trusting.';
        } else if (probability > 40) {
            recommendation = 'This text contains some patterns consistent with AI generation. Consider cross-checking with other sources.';
        } else {
            recommendation = 'This text appears to be primarily human-written. Few AI patterns detected.';
        }
        
        const resultHtml = `
            <div class="result-card">
                <div class="result-header">
                    <h4>Text Analysis Results</h4>
                    <div class="result-score ${scoreClass}">${scoreLabel}: ${probability.toFixed(1)}%</div>
                </div>
                
                <div class="progress-container">
                    <div class="progress-label">
                        <span>AI-Generated Probability</span>
                        <span class="text-${scoreClass}">${probability.toFixed(1)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-${scoreClass}" role="progressbar" 
                             style="width: ${probability}%" 
                             aria-valuenow="${probability}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>
                
                <div class="progress-container">
                    <div class="progress-label">
                        <span>Pattern Matches</span>
                        <span>${Object.keys(data.analysis_details.pattern_matches).length}</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-info" role="progressbar" 
                             style="width: ${Math.min(Object.keys(data.analysis_details.pattern_matches).length * 10, 100)}%" 
                             aria-valuenow="${Math.min(Object.keys(data.analysis_details.pattern_matches).length * 10, 100)}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>

                <div class="result-message">
                    <i class="fas fa-info-circle me-2"></i> ${recommendation}
                </div>
                
                <div class="result-actions">
                    <button class="btn btn-outline-primary" onclick="showDetailedAnalysis(${JSON.stringify(data).replace(/"/g, '&quot;')})">
                        <i class="fas fa-chart-line me-2"></i>Detailed Analysis
                    </button>
                </div>
            </div>`;
        
        textResults.innerHTML = resultHtml;
    }

    // Make function available globally for buttons
    window.showDetailedAnalysis = function(data) {
        const modalContent = document.getElementById('modalContent');
        
        let detailedHtml = `
            <div class="chart-container">
                <canvas id="analysisChart"></canvas>
            </div>`;
        
        if (data.deepfake_analysis) {
            // Extract key metrics for radar chart
            const analysis_details = data.deepfake_analysis.analysis_details;
            
            // Add new metrics from enhanced analysis
            const ela_score = analysis_details.ela_score || 0;
            const ai_generated_prob = analysis_details.ai_generated_probability || 0;
            
            // Image/Video analysis details
            detailedHtml += `
                <div class="analysis-metrics">
                    <h5 class="mt-4 mb-3">Detailed Analysis Metrics</h5>
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                    <th>Interpretation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Faces Detected</td>
                                    <td>${analysis_details.faces_detected}</td>
                                    <td>${interpretFaceCount(analysis_details.faces_detected)}</td>
                                </tr>
                                <tr>
                                    <td>AI Generation Probability</td>
                                    <td>${(ai_generated_prob * 100).toFixed(1)}%</td>
                                    <td>${interpretAIGenerationProbability(ai_generated_prob)}</td>
                                </tr>
                                <tr>
                                    <td>Error Level Analysis</td>
                                    <td>${(ela_score * 100).toFixed(1)}%</td>
                                    <td>${interpretELAScore(ela_score)}</td>
                                </tr>
                                <tr>
                                    <td>Frequency Anomalies</td>
                                    <td>${analysis_details.frequency_anomalies.length > 0 ? 'Present' : 'None detected'}</td>
                                    <td>${interpretFrequencyAnomalies(analysis_details.frequency_anomalies)}</td>
                                </tr>
                                <tr>
                                    <td>Noise Inconsistency</td>
                                    <td>${(analysis_details.noise_inconsistency * 100).toFixed(1)}%</td>
                                    <td>${interpretNoiseLevel(analysis_details.noise_inconsistency)}</td>
                                </tr>
                                <tr>
                                    <td>Compression Artifacts</td>
                                    <td>${(analysis_details.compression_artifacts * 100).toFixed(1)}%</td>
                                    <td>${interpretCompressionArtifacts(analysis_details.compression_artifacts)}</td>
                                </tr>
                                <tr>
                                    <td>AI Artifacts</td>
                                    <td>${analysis_details.ai_artifacts.length || 'None'}</td>
                                    <td>${interpretAIArtifacts(analysis_details.ai_artifacts)}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>`;
            
            // Add radar chart data for new metrics
            chartData = {
                labels: [
                    'CNN Classifier', 
                    'Error Level Analysis', 
                    'Face Analysis', 
                    'Frequency Analysis', 
                    'Noise Analysis',
                    'Compression Artifacts',
                    'AI Artifacts'
                ],
                datasets: [{
                    label: 'Detection Metrics',
                    data: [
                        // CNN classifier score (AI generation probability)
                        ai_generated_prob * 100,
                        // ELA score
                        ela_score * 100,
                        // Face analysis (average face score if faces detected)
                        analysis_details.faces_detected > 0 ? 
                            analysis_details.face_analysis_scores.reduce((sum, score) => sum + score, 0) / 
                            analysis_details.face_analysis_scores.length * 100 : 0,
                        // Frequency anomaly score based on anomalies detected
                        analysis_details.frequency_anomalies.length > 0 ? 70 : 30,
                        // Noise inconsistency
                        analysis_details.noise_inconsistency * 100,
                        // Compression artifacts
                        analysis_details.compression_artifacts * 100,
                        // AI artifacts score
                        analysis_details.ai_artifacts.length * 25
                    ],
                    backgroundColor: 'rgba(53, 99, 233, 0.4)',
                    borderColor: 'rgba(53, 99, 233, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(53, 99, 233, 1)',
                    pointRadius: 4
                }]
            };
        } else {
            // Text analysis details
            const details = data.analysis_details;
            detailedHtml += `
                <div class="analysis-metrics">
                    <h5 class="mt-4 mb-3">Text Analysis Metrics</h5>
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                    <th>Interpretation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>AI Pattern Matches</td>
                                    <td>${Object.keys(details.pattern_matches).length}</td>
                                    <td>${interpretPatternMatches(Object.keys(details.pattern_matches).length)}</td>
                                </tr>
                                <tr>
                                    <td>Complexity Score</td>
                                    <td>${(details.complexity_score * 100).toFixed(1)}%</td>
                                    <td>${interpretComplexity(details.complexity_score)}</td>
                                </tr>
                                <tr>
                                    <td>Repetition Score</td>
                                    <td>${(details.repetition_score * 100).toFixed(1)}%</td>
                                    <td>${interpretRepetition(details.repetition_score)}</td>
                                </tr>
                                <tr>
                                    <td>Filler Word Score</td>
                                    <td>${(details.filler_word_score * 100).toFixed(1)}%</td>
                                    <td>${interpretFillerWords(details.filler_word_score)}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>`;
            
            // Use existing text chart data
        }
        
        modalContent.innerHTML = detailedHtml;
        
        // Create chart
        const ctx = document.getElementById('analysisChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'radar',
            data: chartData,
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            backdropColor: 'transparent'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        angleLines: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        pointLabels: {
                            font: {
                                size: 12,
                                family: "'Inter', sans-serif"
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleFont: {
                            family: "'Inter', sans-serif",
                            size: 14
                        },
                        bodyFont: {
                            family: "'Inter', sans-serif",
                            size: 13
                        },
                        padding: 12,
                        cornerRadius: 8
                    }
                },
                elements: {
                    line: {
                        tension: 0.1
                    }
                }
            }
        });
        
        // Show modal
        resultModal.show();
    };

    // Interpretation helpers
    function interpretFaceCount(count) {
        if (count === 0) return 'No faces detected in the image';
        if (count === 1) return 'Single face detected for analysis';
        return `Multiple faces (${count}) detected for analysis`;
    }

    function interpretFrequencyAnomalies(anomalies) {
        if (anomalies.length === 0) return 'No unusual frequency patterns detected';
        return 'Unusual frequency patterns detected, which may indicate AI generation';
    }

    function interpretNoiseLevel(level) {
        if (level < 0.3) return 'Natural noise distribution';
        if (level < 0.7) return 'Some noise inconsistencies detected';
        return 'Significant noise inconsistencies, suggesting possible manipulation';
    }

    function interpretCompressionArtifacts(level) {
        if (level < 0.3) return 'Normal compression patterns';
        if (level < 0.7) return 'Moderate compression artifacts detected';
        return 'High level of compression artifacts, may hide manipulation traces';
    }

    function interpretAIArtifacts(artifacts) {
        if (artifacts.length === 0) return 'No specific AI artifacts detected';
        return `${artifacts.length} AI-specific artifacts detected, suggesting synthetic generation`;
    }

    function interpretPatternMatches(count) {
        if (count < 3) return 'Few AI patterns detected, likely human-written';
        if (count < 7) return 'Moderate number of AI patterns detected';
        return 'High number of AI patterns, strongly suggesting AI generation';
    }

    function interpretComplexity(score) {
        if (score < 0.3) return 'Low complexity, typical of simple AI-generated text';
        if (score < 0.7) return 'Moderate complexity';
        return 'High complexity, suggesting either sophisticated AI or human authorship';
    }

    function interpretRepetition(score) {
        if (score < 0.3) return 'Low repetition, typical of human writing';
        if (score < 0.7) return 'Moderate repetition detected';
        return 'High repetition, typical of AI-generated content';
    }

    function interpretFillerWords(score) {
        if (score < 0.3) return 'Few filler words, typical of human writing';
        if (score < 0.7) return 'Moderate use of filler words';
        return 'High use of filler words, common in AI-generated text';
    }
    
    // New interpretation functions
    function interpretELAScore(score) {
        if (score < 0.3) return 'Low ELA score, suggesting authentic image';
        if (score < 0.6) return 'Moderate ELA score, some editing may be present';
        return 'High ELA score, indicating significant editing or AI generation';
    }

    function interpretAIGenerationProbability(probability) {
        if (probability < 0.3) return 'Low probability of AI generation';
        if (probability < 0.7) return 'Moderate probability of AI generation';
        return 'High probability that this image was AI-generated';
    }

    // Download report functionality
    document.getElementById('downloadReport').addEventListener('click', function() {
        alert('Report download functionality will be available in a future update.');
    });
}); 