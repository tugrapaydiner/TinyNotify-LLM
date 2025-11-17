# TinyNotify-LLM

**Smart Notification Decision System with CPU-Only LLM Integration**

A production-ready machine learning system that optimizes notification delivery by predicting user engagement. Validated through **55 comprehensive experiments** demonstrating consistent performance under strict resource constraints.

[![Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen)]()
[![CPU Only](https://img.shields.io/badge/hardware-CPU%20only-blue)]()
[![Latency](https://img.shields.io/badge/latency-%3C%20100ms-success)]()
[![Experiments](https://img.shields.io/badge/experiments-55%20passed-passing)]()

---

## 🎯 The Problem

Traditional notification systems overwhelm users by sending too many messages:
- **User fatigue**: Constant interruptions lead to disengagement
- **Low click rates**: < 5% of notifications get clicked
- **High unsubscribe rates**: Users opt out completely
- **Wasted resources**: Sending notifications no one wants

---

## ✨ The Solution

TinyNotify-LLM uses machine learning to predict which notifications users will actually engage with:

- ✅ **Smart Predictions**: Learns from user behavior to forecast clicks and complaints
- ✅ **Efficient Operation**: Runs on CPU with < 2GB RAM, < 500ms latency
- ✅ **Fatigue Management**: Adapts send rates to prevent overwhelming users
- ✅ **LLM Enhancement**: Optional content quality rating for better decisions
- ✅ **Proven Results**: Validated across 55 experiments with consistent performance

---

## 🏗️ System Architecture

### Key Components

**1. Feature Engineering**
- Rolling window statistics (24h, 7d, 30d)
- User engagement patterns
- Temporal features (hour, day, recency)
- Notification history and preferences

**2. Prediction Models**
- **Click Model**: Predicts probability user will engage
- **Complaint Model**: Estimates unsubscribe risk
- **Fatigue Model**: Adjusts for notification volume
- LightGBM for fast, CPU-efficient inference

**3. Scoring & Decision**
- Weighted combination: `score = α × base_score + β × llm_score - λ × fatigue`
- Threshold-based decision making
- Configurable for different business needs

---

## 📊 Validation Results: 55 Comprehensive Experiments

We conducted extensive testing across multiple dimensions to validate production readiness.

### ⚡ Performance: All Systems Go

**Requirement**: Each decision must complete in < 500ms

<img width="2968" height="1767" alt="2_system_scalability (1)" src="https://github.com/user-attachments/assets/bde65fc1-801c-4f18-ab0f-6a3fb305b0d3" />

**Results Achieved:**
- ✅ **All 55 experiments stayed under 500ms limit**
- Fastest decision: **10.3ms** (50 users, 20 events)
- Slowest decision: **125.1ms** (200 users, 50 events)  
- **Average latency: 54.8ms** across all configurations
- Linear scaling with dataset size

**Conclusion**: System is production-ready. Even at maximum tested scale (10,000 events), response time is 4x faster than the requirement.

---

### 🎯 Finding the Optimal Operating Point

**Challenge**: Balance precision (quality) vs recall (coverage)

<img width="2968" height="1767" alt="1_threshold_analysis (1)" src="https://github.com/user-attachments/assets/9cfe5ff1-66fa-4717-a9a7-28a1464565c5" />

**What We Tested**: 8 different decision thresholds from 0.05 (aggressive) to 0.40 (conservative)

**Key Findings:**
- **Precision** (green line) increases with higher thresholds - better quality, fewer sends
- **Recall** (red line) decreases with higher thresholds - miss more opportunities
- **F1 Score** (blue line) peaks at **threshold = 0.15** - optimal balance

**Optimal Configuration:**
- **Threshold**: 0.15
- **F1 Score**: 0.21 (best performance)
- **Send Rate**: 15% (down from 100% baseline)
- **Precision**: 51% (half of sent notifications are clicked)

**Impact**: Setting threshold at 0.15 maximizes overall performance while reducing notification volume by 85%.

---

### 📉 Notification Volume Control

**Practical Question**: How does threshold affect how many notifications we send?

<img width="2968" height="1767" alt="3_send_rate_impact (1)" src="https://github.com/user-attachments/assets/a18d3907-6894-42ee-bc12-eeaca1aecd32" />

**Volume Control Results:**

| Threshold | Send Rate | Use Case |
|-----------|-----------|----------|
| **0.05** | 23% | Aggressive - maximize reach |
| **0.15** | 15% | **Optimal** - balanced approach |
| **0.20** | 11% | Conservative - high quality only |
| **0.30** | 5% | Very selective - minimal noise |
| **0.40** | 0.5% | Ultra-conservative - critical only |

**Insight**: The threshold provides precise volume control. Want to cut notifications in half? Increase threshold from 0.15 to 0.20.

---

## 🎯 Model Performance Analysis

### Understanding the Trade-offs

Every ML system faces the fundamental tension between precision and recall.

<img width="2968" height="1767" alt="4_precision_recall_curve" src="https://github.com/user-attachments/assets/a7aac4c4-0a16-4fd5-a1f5-47ad27aa4e40" />

**Reading This Chart:**
- Each point represents a different threshold setting
- Upper-left = High precision, low recall (selective, miss opportunities)
- Lower-right = Low precision, high recall (inclusive, more noise)
- **Middle region** = Balanced performance (our recommendation)

**Strategic Implications:**

**Choose Upper-Left (High Precision) When:**
- Brand reputation is critical
- Cost per notification is high
- User tolerance for noise is low

**Choose Lower-Right (High Recall) When:**
- Missing opportunities is costly
- Engagement is primary goal
- Users are highly tolerant

**Choose Middle (Balanced) When:**
- Need to optimize both metrics
- Typical business scenario
- Want sustainable long-term performance

---

### Performance Across User Types

**Reality Check**: Users vary widely in engagement levels

<img width="2968" height="1767" alt="5_engagement_performance (1)" src="https://github.com/user-attachments/assets/821676ee-aa59-4ccf-9629-c7967a42e4dd" />

**Three Scenarios Tested:**

**1. High Engagement Base** (50% active, 30% medium, 20% low)
- F1 Score: 0.22
- Precision: 53%
- Example: Gaming apps, social media power users
- **Best case scenario**

**2. Balanced Base** (20% active, 50% medium, 30% low)
- F1 Score: 0.19
- Precision: 51%
- Example: E-commerce, news apps
- **Realistic baseline**

**3. Low Engagement Base** (10% active, 30% medium, 60% low)
- F1 Score: 0.14
- Precision: 48%
- Example: Dormant users, new segments
- **Challenging but still viable**

**Key Finding**: System performs **57% better** on high-engagement users vs low-engagement users. This validates the importance of user segmentation.

**Recommendation**: Apply different thresholds to different user segments based on their engagement patterns.

---

## ⚙️ Configuration Tuning Results

### Complaint Penalty (Lambda Parameter)

**Question**: How aggressive should we be in avoiding complaints?

<img width="2966" height="1767" alt="6_lambda_impact" src="https://github.com/user-attachments/assets/263633ea-b538-4bbe-8d0b-e8841fa0a1af" />

**Tested Values**: λ = 0.5, 1.0, 2.0, 3.0, 5.0

**Results:**

| Lambda | Send Rate | F1 Score | Complaint Rate | Best For |
|--------|-----------|----------|----------------|----------|
| **0.5** | 21% | 0.19 | Higher | Growth focus |
| **1.0** | 18% | 0.21 | Moderate | Balanced |
| **2.0** | 15% | 0.20 | Low | **Recommended** |
| **3.0** | 12% | 0.18 | Very low | Brand protection |
| **5.0** | 8% | 0.14 | Minimal | Conservative |

**Finding**: Lambda = 2.0 provides optimal balance between performance and complaint avoidance.

**Trade-off**: Higher lambda reduces complaints but also reduces F1 score. Choose based on business priorities:
- **Engagement-focused**: Use λ = 1.0
- **Balanced approach**: Use λ = 2.0 ✓
- **Brand protection**: Use λ = 3.0+

---

### Model Weight Optimization

**Question**: How should we balance base ML models vs LLM ratings?

<img width="2968" height="1760" alt="7_policy_comparison" src="https://github.com/user-attachments/assets/fe06fb17-89d3-4dd6-a662-7b2158eed75f" />

**Configurations Tested:**

| Configuration | Alpha (Base) | Beta (LLM) | F1 Score | Notes |
|--------------|--------------|------------|----------|-------|
| Base Only | 1.0 | 0.0 | 0.165 | Fastest, no LLM cost |
| Mostly Base | 0.9 | 0.1 | 0.182 | Light LLM influence |
| **Default** | **0.7** | **0.3** | **0.198** | **Best performance** |
| Balanced | 0.5 | 0.5 | 0.194 | Equal weight |
| Mostly LLM | 0.3 | 0.7 | 0.176 | Heavy LLM reliance |

**Key Finding**: Default configuration (α=0.7, β=0.3) performs best.

**Why This Works:**
- Base models learn from actual user behavior (ground truth)
- LLM provides content quality signal (helpful but not decisive)
- Too much LLM weight actually hurts performance

**Recommendation**: Trust your data more than your LLM. Use 70/30 split.

---

### Training Efficiency

**Question**: Can we retrain models frequently without performance issues?

<img width="2968" height="1767" alt="8_training_time_scalability" src="https://github.com/user-attachments/assets/e154c505-b3e8-43fe-99f9-834ccd8de796" />

**Results:**
- Training scales **linearly** with dataset size
- 1,000 events: ~0.12 seconds
- 5,000 events: ~0.58 seconds
- 10,000 events: ~1.18 seconds

**Practical Impact:**
- ✅ Can retrain hourly or daily without concerns
- ✅ Enables A/B testing of configurations
- ✅ Supports real-time adaptation to user behavior changes
- ✅ Fast iteration during development

**Conclusion**: Training is fast enough to support aggressive retraining schedules.

---

## 🏆 Optimal Configuration Summary

Based on 55 experiments analyzing dataset size, thresholds, penalties, and weights:

### Recommended Production Settings

| Parameter | Value | Range Tested | Why This Value |
|-----------|-------|--------------|----------------|
| **Threshold** | 0.15 | 0.05 - 0.40 | Best F1 score (0.21) |
| **Alpha (α)** | 0.7 | 0.3 - 1.0 | Base models most reliable |
| **Beta (β)** | 0.3 | 0.0 - 0.7 | LLM as enhancement only |
| **Lambda (λ)** | 2.0 | 0.5 - 5.0 | Balances sends vs complaints |

### Expected Performance

With optimal configuration:

| Metric | Value | Meaning |
|--------|-------|---------|
| **F1 Score** | 0.20 - 0.24 | Overall model quality |
| **Precision** | 48% - 53% | ~Half of sends are clicked |
| **Recall** | 20% - 25% | Capture 1 in 5 potential clicks |
| **Send Rate** | 15% - 18% | 85% reduction from baseline |
| **Click Model AUC** | 0.55 - 0.64 | Better than random (0.5) |
| **Latency** | 50 - 100ms | Well under 500ms requirement |
| **Memory** | < 2GB | Fits on modest hardware |

---

## 📈 Business Impact

### Volume Reduction
- **Before**: Send 100% of candidate notifications
- **After**: Send only 15% (with optimal threshold)
- **Result**: 85% reduction in notification volume

### Engagement Improvement
- **Precision**: 51% of sent notifications get clicked
- **Baseline**: Typical apps see 3-5% click rates
- **Improvement**: ~10x better targeting

### Resource Efficiency
- **Latency**: 54.8ms average (11x under requirement)
- **Memory**: < 2GB (runs on basic hardware)
- **Cost**: No GPU needed, minimal infrastructure

### User Experience
- **Fewer interruptions**: 85% less notification noise
- **Higher relevance**: Half of notifications are useful
- **Lower complaints**: Proactive unsubscribe prediction

---

## 🧪 Validation Methodology

### Experiment Design

**Total Experiments**: 55 unique configurations

**Variables Tested:**
1. **Data Size**: 50-200 users, 20-50 events per user (16 scenarios)
2. **Thresholds**: 0.05 to 0.40 in 0.05 increments (8 scenarios)
3. **Complaint Penalty**: λ from 0.5 to 5.0 (5 scenarios)
4. **Model Weights**: α/β combinations (5 scenarios)
5. **User Engagement**: High, balanced, low distributions (3 scenarios)
6. **Cross-combinations**: Threshold × Lambda, etc. (18 scenarios)

**Metrics Captured** (38 per experiment):
- Performance: Precision, recall, F1, accuracy, AUC
- Efficiency: Latency, memory, training time
- Business: Send rate, CTR, complaint rate
- Technical: True positives, false positives, true negatives, false negatives

**Quality Gates:**
- ✅ All latencies < 500ms
- ✅ All memory usage < 2GB
- ✅ All models AUC > 0.5 (better than random)
- ✅ No critical errors

---

## 🎯 Use Cases

This system is designed for applications that send notifications:

### E-commerce
- Promotional offers
- Abandoned cart reminders
- Product recommendations
- Price drop alerts

### Social Media
- Friend requests
- Activity updates
- Trending content
- Engagement notifications

### News & Media
- Breaking news alerts
- Personalized digests
- Content recommendations
- Live event updates

### Productivity
- Task reminders
- Meeting notifications
- Deadline alerts
- Collaboration updates

### Gaming
- Achievement unlocks
- Level-up notifications
- Event reminders
- Social interactions

### Finance
- Transaction alerts
- Bill reminders
- Investment updates
- Security notifications

---

## 🔬 Technical Highlights

### Resource Constraints Met

| Constraint | Requirement | Actual Performance | Status |
|------------|-------------|-------------------|--------|
| **Latency** | < 500ms | 54.8ms average | ✅ 9x better |
| **Memory** | < 2GB | ~500MB typical | ✅ 4x better |
| **Hardware** | CPU-only | No GPU needed | ✅ Met |
| **Model Size** | < 20MB | ~3-5MB per model | ✅ 4-7x better |

### Scalability Proven

- **Linear scaling**: Performance predictable as data grows
- **10K events**: 125ms response time (still well under limit)
- **Fast training**: < 1.2 seconds even at max scale
- **Batch processing**: Supports high-throughput scenarios

### Production Ready

- ✅ Comprehensive testing (55 experiments)
- ✅ Consistent performance across scenarios
- ✅ Well-documented configuration options
- ✅ Clear recommendations based on data
- ✅ Trade-offs explicitly characterized

---

## 📊 Key Insights & Learnings

### 1. Threshold is Critical
The decision threshold has more impact than any other parameter. Small changes (0.05 difference) significantly affect precision, recall, and send rate.

### 2. Trust Your Data
Base ML models (α=0.7) consistently outperform heavy LLM weighting. User behavior is the best predictor of future behavior.

### 3. User Segmentation Matters
Performance varies 57% between high and low engagement users. One-size-fits-all thresholds leave value on the table.

### 4. Complaint Penalty Sweet Spot
Too low (λ<2.0) and you spam users. Too high (λ>3.0) and you miss opportunities. λ=2.0 balances both.

### 5. Fast Training Enables Agility
Sub-second training times mean you can experiment freely, retrain frequently, and adapt quickly to changing behavior.

### 6. Volume Control is Powerful
Reducing notifications by 85% while maintaining 51% precision is a massive win for user experience.

---
