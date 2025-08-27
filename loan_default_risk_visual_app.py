# loan_default_visual_app_enhanced.py

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page settings
st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="wide")

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d3748 100%);
    }
    
    .section-header {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4f46e5;
        margin: 1.5rem 0 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .section-header h3 {
        margin: 0;
        color: #ffffff;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    .stNumberInput > label, .stSelectbox > label {
        color: #e2e8f0 !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff;
    }
    
    .title-container {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("logistic_loan_default.pkl"):
            st.error("⚠️ Model file 'logistic_loan_default.pkl' not found. Please ensure the model file is in the correct directory.")
            st.info("📋 Expected file: logistic_loan_default.pkl")
            st.stop()
        model = joblib.load("logistic_loan_default.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# --- HEADER IMAGE ---
st.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3", use_container_width=True)

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2331/2331941.png", width=80)
st.sidebar.title("📊 Risk Assessment Tool")
st.sidebar.markdown("""
**How it works:**
1. Enter client and loan details
2. AI model analyzes risk factors
3. Get instant credit score & recommendation

**Risk Factors Considered:**
- Payment history
- Loan characteristics  
- Employment status
- Banking relationship
""")

# Feature importance for educational purposes
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Key Risk Indicators:**")
st.sidebar.markdown("""
- **Payment History**: Most important factor
- **Loan Amount vs Income**: Debt-to-income ratio
- **Previous Defaults**: Past behavior predictor
- **Employment Stability**: Income reliability
""")

# --- HEADER ---
st.markdown("""
<div class="title-container">
    <h1 style="color: #ffffff; margin: 0; font-size: 2.5rem;">🏦 Loan Default Risk Predictor</h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Assess loan repayment probability using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p style="color: #94a3b8; text-align: center; font-style: italic;">All fields are required for accurate risk assessment</p>', unsafe_allow_html=True)

# --- INPUT SECTIONS ---

# Loan Details Section
st.markdown("""
<div class="section-header">
    <h3>💰 Loan Details</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    loanamount = st.number_input("Loan Amount", min_value=100, max_value=1000000, value=50000)
    termdays = st.number_input("Loan Term (days)", 10, 720, 90)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    repayment_curr_ratio = st.number_input("Repayment Current Ratio", 0.0, 2.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Payment History Section
st.markdown("""
<div class="section-header">
    <h3>📊 Payment History</h3>
</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    num_prev_loans = st.number_input("Number of Previous Loans", 0.00, 50.00, 3.00)
    avg_repay_delay_days = st.number_input("Average Repay Delay (days)", -50.00, 365.00, 10.00)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    total_firstrepaid_late = st.number_input("Total First Repaid Late", 0.00, 50.00, 2.00)
    st.markdown('</div>', unsafe_allow_html=True)

# Financial History Section
st.markdown("""
<div class="section-header">
    <h3>📈 Financial History</h3>
</div>
""", unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    avg_prev_repayment_ratio = st.number_input("Avg Previous Repayment Ratio", 0.0, 2.0, 1.0)
    avg_duration_days = st.number_input("Avg Duration of Previous Loans (days)", 0.00, 720.00, 180.00)
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    avg_prev_interest = st.number_input("Avg Previous Interest", 0.00, 100000.00, 5000.00)
    age = st.number_input("Client Age", 18, 100, 30)
    st.markdown('</div>', unsafe_allow_html=True)

# Banking & Employment Profile Section
st.markdown("""
<div class="section-header">
    <h3>🏦 Banking & Employment Profile</h3>
</div>
""", unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    bank_account_type = st.selectbox("Bank Account Type", ['Other', 'Savings', 'Current'])
    st.markdown('</div>', unsafe_allow_html=True)

with col8:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    employment_status_clients = st.selectbox(
        "Employment Status",
        ['Permanent', 'Unknown', 'Unemployed', 'Self-Employed', 'Student', 'Retired', 'Contract']
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Input validation
def validate_inputs():
    errors = []
    warnings = []
    
    if loanamount <= 0:
        errors.append("Loan amount must be positive")
    
    if repayment_curr_ratio < 0.1:
        warnings.append("Very low repayment ratio may indicate high risk")
    elif repayment_curr_ratio < 0.5:
        warnings.append("Low repayment ratio detected")
    
    if avg_repay_delay_days > 30:
        warnings.append("High average payment delay may affect approval")
    
    if num_prev_loans > 0 and total_firstrepaid_late / num_prev_loans > 0.5:
        warnings.append("High rate of late first payments detected")
    
    return errors, warnings

# --- Feature Engineering ---
try:
    repayment_efficiency = repayment_curr_ratio / (avg_prev_repayment_ratio + 1e-6)
    late_payment_rate = total_firstrepaid_late / (num_prev_loans + 1e-6) if num_prev_loans > 0 else 0

    sqrt_loanamount = np.sqrt(loanamount)
    sqrt_termdays = np.sqrt(termdays)
    sqrt_avg_prev_interest = np.sqrt(avg_prev_interest)
    sqrt_repayment_efficiency = np.sqrt(abs(repayment_efficiency))
    sqrt_late_payment_rate = np.sqrt(late_payment_rate)
    
except Exception as e:
    st.error(f"Error in feature calculation: {str(e)}")
    st.stop()

# --- PREDICT BUTTON ---
st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
col_predict, col_clear = st.columns([3, 1])

with col_predict:
    predict_button = st.button("🚀 Predict Loan Risk", use_container_width=True, type="primary")

with col_clear:
    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

if predict_button:
    # Validate inputs
    errors, warnings = validate_inputs()
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    else:
        # Show warnings if any
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
        
        try:
            # Prepare data
            data = {
                "loanamount": [loanamount],
                "termdays": [termdays],
                "repayment_curr_ratio": [repayment_curr_ratio],
                "num_prev_loans": [num_prev_loans],
                "avg_repay_delay_days": [avg_repay_delay_days],
                "total_firstrepaid_late": [total_firstrepaid_late],
                "avg_prev_repayment_ratio": [avg_prev_repayment_ratio],
                "avg_duration_days": [avg_duration_days],
                "avg_prev_interest": [avg_prev_interest],
                "age": [age],
                "bank_account_type": [bank_account_type],
                "employment_status_clients": [employment_status_clients],
                "repayment_efficiency": [repayment_efficiency],
                "late_payment_rate": [late_payment_rate],
                "sqrt_loanamount": [sqrt_loanamount],
                "sqrt_termdays": [sqrt_termdays],
                "sqrt_avg_prev_interest": [sqrt_avg_prev_interest],
                "sqrt_repayment_efficiency": [sqrt_repayment_efficiency],
                "sqrt_late_payment_rate": [sqrt_late_payment_rate]
            }
            df = pd.DataFrame(data)

            # Predict
            proba_good = model.predict_proba(df)[0, 1]
            proba_bad = 1 - proba_good
            
            # Credit score calculation
            min_score, max_score = 300, 850
            credit_score = min_score + (max_score - min_score) * proba_good
            good_threshold = 575
            classification = "Good" if credit_score >= good_threshold else "Bad"
            
            # Risk level categorization
            if credit_score >= 750:
                risk_level = "Excellent"
                risk_color = "#10b981"
                risk_icon = "🟢"
            elif credit_score >= 700:
                risk_level = "Good"
                risk_color = "#22c55e" 
                risk_icon = "🟢"
            elif credit_score >= 650:
                risk_level = "Fair"
                risk_color = "#f59e0b"
                risk_icon = "🟡"
            elif credit_score >= 575:
                risk_level = "Poor"
                risk_color = "#f97316"
                risk_icon = "🟠"
            else:
                risk_level = "Very Poor"
                risk_color = "#ef4444"
                risk_icon = "🔴"

            # --- RESULTS SECTION ---
            st.markdown("""
            <div class="prediction-container">
                <h2 style="color: #ffffff; text-align: center; margin-bottom: 2rem;">📊 Risk Assessment Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary metrics
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            
            with col_summary1:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                    <h3 style="color: #ffffff; margin: 0; font-size: 2rem;">{credit_score:.0f}</h3>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Credit Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_summary2:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                    <h3 style="color: {risk_color}; margin: 0; font-size: 1.5rem;">{risk_icon} {risk_level}</h3>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Risk Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_summary3:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                    <h3 style="color: #ef4444; margin: 0; font-size: 2rem;">{proba_bad:.1%}</h3>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Default Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_summary4:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                    <h3 style="color: #10b981; margin: 0; font-size: 2rem;">{proba_good:.1%}</h3>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Repayment Probability</p>
                </div>
                """, unsafe_allow_html=True)

            # Charts
            st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                # Enhanced gauge chart
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=credit_score,
                    domain={'x': [0, 1], 'y': [0.1, 0.9]},
                    title={'text': f"Credit Score", 'font': {'size': 24, 'color': '#ffffff'}},
                    number={'font': {'size': 48, 'color': '#ffffff'}},
                    gauge={
                        'axis': {'range': [300, 850], 'tickwidth': 2, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#94a3b8'}},
                        'bar': {'color': risk_color, 'thickness': 0.3},
                        'bgcolor': "rgba(255,255,255,0.1)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.3)",
                        'steps': [
                            {'range': [300, 575], 'color': 'rgba(239, 68, 68, 0.3)'},
                            {'range': [575, 650], 'color': 'rgba(245, 158, 11, 0.3)'},
                            {'range': [650, 750], 'color': 'rgba(34, 197, 94, 0.3)'},
                            {'range': [750, 850], 'color': 'rgba(16, 185, 129, 0.5)'}
                        ],
                        'threshold': {
                            'line': {'color': "#ef4444", 'width': 4},
                            'thickness': 0.8,
                            'value': 575
                        }
                    }
                ))
                gauge.update_layout(
                    height=500, 
                    margin=dict(l=30, r=30, t=100, b=50),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#ffffff'}
                )
                st.plotly_chart(gauge, use_container_width=True)

            with col2:
                # Risk factors contribution
                risk_factors = {
                    'Payment History': min(max(20 + avg_repay_delay_days * 2 + total_firstrepaid_late * 5, 5), 45),
                    'Repayment Capacity': min(max((2 - repayment_curr_ratio) * 15, 5), 35),
                    'Loan Characteristics': min(max(loanamount / 10000 + termdays / 50, 5), 25),
                    'Employment Status': 15 if employment_status_clients == 'Permanent' else 25,
                    'Previous Performance': min(max(late_payment_rate * 30, 5), 30)
                }
                
                factor_chart = go.Figure()
                factor_chart.add_trace(go.Bar(
                    y=list(risk_factors.keys()),
                    x=list(risk_factors.values()),
                    orientation='h',
                    marker_color=[risk_color if v > 25 else '#f59e0b' if v > 15 else '#10b981' for v in risk_factors.values()],
                    text=[f"{v:.1f}%" for v in risk_factors.values()],
                    textposition='inside',
                    textfont={'color': '#ffffff', 'size': 12}
                ))
                
                factor_chart.update_layout(
                    title={'text': "Risk Factor Analysis", 'font': {'size': 24, 'color': '#ffffff'}},
                    xaxis_title="Risk Impact (%)",
                    yaxis_title="",
                    height=500,
                    margin=dict(l=20, r=20, t=100, b=50),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis={'color': '#94a3b8', 'gridcolor': 'rgba(255,255,255,0.1)'},
                    yaxis={'color': '#94a3b8'},
                    font={'color': '#ffffff'}
                )
                
                st.plotly_chart(factor_chart, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Final Recommendation
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            
            if classification == "Good":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10b981, #059669); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h2 style="color: #ffffff; margin: 0;">✅ LOAN APPROVED</h2>
                    <h3 style="color: #d1fae5; margin: 0.5rem 0;">Low Default Risk</h3>
                    <p style="color: #ffffff; margin: 1rem 0; font-size: 1.1rem;">
                        <strong>Credit Score:</strong> {credit_score:.0f}/850 ({risk_level})<br>
                        <strong>Default Probability:</strong> {proba_bad:.1%}<br>
                        <strong>Recommendation:</strong> Approve with standard terms
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ef4444, #dc2626); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h2 style="color: #ffffff; margin: 0;">❌ LOAN DECLINED</h2>
                    <h3 style="color: #fecaca; margin: 0.5rem 0;">High Default Risk</h3>
                    <p style="color: #ffffff; margin: 1rem 0; font-size: 1.1rem;">
                        <strong>Credit Score:</strong> {credit_score:.0f}/850 ({risk_level})<br>
                        <strong>Default Probability:</strong> {proba_bad:.1%}<br>
                        <strong>Recommendation:</strong> Decline or require additional collateral
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Timestamp
            st.markdown(f"""
            <div style="text-align: center; color: #64748b; margin-top: 2rem; font-size: 0.9rem;">
                Assessment completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check your inputs and try again. If the problem persists, contact support.")

# Footer
st.markdown("""
<div style='text-align: center; color: #64748b; margin-top: 3rem; padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);'>
    <small>
    👩‍💻 Built by: <strong>Samson Chigozie Chigbu</strong><br>
    📧 Email: <a href="mailto:samsonchigbu5@gmail.com" style="color:#94a3b8;">samsonchigbu5@gmail.com</a><br>
    💼 LinkedIn: <a href="https://www.linkedin.com/in/samson-chigbu-15a0051b4" target="_blank" style="color:#94a3b8;">samson-chigbu-15a0051b4</a>
    </small>
    <br><br>
    <small>
    ⚠️ This tool provides risk assessment for informational purposes only.<br>
    Final lending decisions should consider additional factors and comply with applicable regulations.
    </small>
    
</div>
""", unsafe_allow_html=True)
