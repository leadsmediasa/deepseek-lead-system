"""
DeepSeek Lead Generation Dashboard
Streamlit-based dashboard for lead management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="DeepSeek Lead Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'leads' not in st.session_state:
    st.session_state.leads = []
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = "deepseek"
if 'generated_count' not in st.session_state:
    st.session_state.generated_count = 0

def main():
    """Main dashboard function"""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🤖 DeepSeek Lead Generation System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Lead Intelligence Platform v2026.1.0</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/FFFFFF?text=DeepSeek+AI", use_column_width=True)
        
        st.markdown("### 🎯 Lead Generation")
        
        lead_type = st.selectbox(
            "Lead Type",
            ["Forex", "Crypto", "Casino"],
            index=0
        )
        
        category = st.selectbox(
            "Category",
            ["Hot Leads", "Recovery Leads", "X-Depositor", "New Trader", "Online Scraper"],
            index=0
        )
        
        country = st.multiselect(
            "Country",
            ["DE", "CH", "NL", "AU", "AT", "IT", "SE", "NO", "DK", 
             "CA", "UK", "US", "UAE", "SA", "BH", "JP", "IN", "SG", "MY", "FR", "BG"],
            default=["US", "DE", "UK"]
        )
        
        count = st.slider(
            "Number of Leads",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        ai_model = st.radio(
            "AI Model",
            ["DeepSeek AI", "GPT-4", "Hybrid Model"],
            index=0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            generate_btn = st.button("🚀 Generate Leads", type="primary", use_container_width=True)
        with col2:
            if st.button("📊 Refresh Analytics", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        auto_enrich = st.toggle("Auto-Enrich with AI", value=True)
        validate_leads = st.toggle("Validate Leads", value=True)
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "Excel", "Database"],
            index=0
        )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "🎯 Lead Generator", 
        "📊 Analytics", 
        "⚙️ Settings"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        if generate_btn:
            generate_leads(lead_type.lower(), category.lower().replace(" ", "_"), country, count, ai_model)
        display_lead_generator()
    
    with tab3:
        display_analytics()
    
    with tab4:
        display_settings()

def display_dashboard():
    """Display main dashboard"""
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Leads", "12,548", "+1,234")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Quality Score", "0.78", "+0.02")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Conversion Rate", "15.2%", "+1.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("AI Accuracy", "92.3%", "+0.8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Leads by Country")
        
        # Sample data
        country_data = pd.DataFrame({
            'Country': ['US', 'DE', 'UK', 'CA', 'AU', 'UAE', 'SA', 'JP'],
            'Leads': [2500, 1800, 1500, 1200, 900, 800, 700, 600],
            'Quality': [0.82, 0.79, 0.81, 0.76, 0.75, 0.83, 0.78, 0.77]
        })
        
        fig = px.bar(country_data, x='Country', y='Leads',
                     color='Quality', color_continuous_scale='Viridis',
                     title="Lead Distribution by Country")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Leads by Type")
        
        type_data = pd.DataFrame({
            'Type': ['Forex', 'Crypto', 'Casino'],
            'Count': [5000, 4500, 3000],
            'Conversion': [0.18, 0.14, 0.12]
        })
        
        fig = px.pie(type_data, values='Count', names='Type',
                     title="Lead Distribution by Type",
                     hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent leads table
    st.subheader("🆕 Recent Leads")
    
    # Sample recent leads
    recent_leads = pd.DataFrame({
        'ID': [f'LEAD_{i:06d}' for i in range(10001, 10011)],
        'Name': [f'John Smith {i}' for i in range(10)],
        'Email': [f'john.smith{i}@example.com' for i in range(10)],
        'Country': ['US', 'DE', 'UK', 'CA', 'AU', 'UAE', 'SA', 'JP', 'SG', 'MY'],
        'Type': ['Forex', 'Crypto', 'Casino'] * 3 + ['Forex'],
        'Category': ['Hot', 'Recovery', 'X-Depositor'] * 3 + ['Hot'],
        'AI Score': [0.85, 0.92, 0.78, 0.88, 0.91, 0.79, 0.84, 0.87, 0.90, 0.83],
        'Status': ['New'] * 10
    })
    
    st.dataframe(
        recent_leads,
        use_container_width=True,
        column_config={
            "AI Score": st.column_config.ProgressColumn(
                "AI Score",
                help="AI Quality Score",
                format="%.2f",
                min_value=0,
                max_value=1
            )
        }
    )

def generate_leads(lead_type, category, countries, count, ai_model):
    """Generate leads"""
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate generation
    for i in range(count):
        progress = (i + 1) / count
        progress_bar.progress(progress)
        status_text.text(f"Generating lead {i + 1} of {count}...")
        
        # In production, this would call the actual generator
        # For now, simulate with a delay
    
    progress_bar.empty()
    status_text.empty()
    
    # Show success message
    st.success(f"✅ Successfully generated {count} {lead_type} {category} leads!")
    
    # Update session state
    st.session_state.generated_count += count

def display_lead_generator():
    """Display lead generator interface"""
    
    st.subheader("🎯 Generate New Leads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Configure lead generation parameters in the sidebar and click 'Generate Leads'")
        
        # Show generation stats
        st.metric("Leads Generated Today", st.session_state.generated_count)
        
        if st.session_state.leads:
            st.metric("Current Batch", len(st.session_state.leads))
        
        # Quick actions
        if st.button("🔄 Generate Sample Batch (10 leads)"):
            # Generate sample leads
            sample_leads = generate_sample_leads()
            st.session_state.leads.extend(sample_leads)
            st.success("Generated 10 sample leads!")
    
    with col2:
        st.markdown("### 📤 Export Options")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("💾 Save to CSV"):
                if st.session_state.leads:
                    save_to_csv()
                    st.success("Leads saved to CSV!")
                else:
                    st.warning("No leads to export")
        
        with export_col2:
            if st.button("📄 Save to JSON"):
                if st.session_state.leads:
                    save_to_json()
                    st.success("Leads saved to JSON!")
                else:
                    st.warning("No leads to export")
        
        if st.button("🔄 Clear All Leads"):
            st.session_state.leads = []
            st.session_state.generated_count = 0
            st.success("All leads cleared!")

def display_analytics():
    """Display analytics"""
    
    st.subheader("📊 Advanced Analytics")
    
    # Time period selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Year to date", "All time"],
            index=1
        )
    
    with col2:
        lead_type_filter = st.multiselect(
            "Filter by Type",
            ["Forex", "Crypto", "Casino"],
            default=["Forex", "Crypto", "Casino"]
        )
    
    with col3:
        metric_type = st.selectbox(
            "Metric",
            ["Quantity", "Quality Score", "Conversion Rate", "Revenue"],
            index=0
        )
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality trend chart
        st.subheader("📈 Quality Trend")
        
        # Sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        quality_data = pd.DataFrame({
            'Date': dates,
            'Forex': [0.75 + 0.1 * (i/30) + 0.05 * (i % 7)/7 for i in range(30)],
            'Crypto': [0.70 + 0.15 * (i/30) + 0.04 * (i % 7)/7 for i in range(30)],
            'Casino': [0.65 + 0.12 * (i/30) + 0.03 * (i % 7)/7 for i in range(30)]
        })
        
        fig = px.line(quality_data, x='Date', y=['Forex', 'Crypto', 'Casino'],
                      title="Average Quality Score Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AI model performance
        st.subheader("🤖 AI Model Performance")
        
        model_data = pd.DataFrame({
            'Model': ['DeepSeek AI', 'GPT-4', 'Hybrid', 'Ensemble'],
            'Accuracy': [0.92, 0.89, 0.94, 0.96],
            'Speed (leads/sec)': [45, 32, 38, 28],
            'Cost per 1000': [12.5, 15.0, 14.0, 16.5]
        })
        
        fig = px.bar(model_data, x='Model', y='Accuracy',
                     title="Model Accuracy Comparison",
                     color='Speed (leads/sec)',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📋 Detailed Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Hot Leads", "4,258", "+8.2%")
    with metrics_col2:
        st.metric("Recovery Leads", "3,142", "+12.5%")
    with metrics_col3:
        st.metric("X-Depositors", "2,847", "+5.7%")
    with metrics_col4:
        st.metric("New Traders", "2,301", "+9.3%")

def display_settings():
    """Display settings"""
    
    st.subheader("⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["AI Configuration", "Country Settings", "Export Settings"])
    
    with tab1:
        st.markdown("### 🤖 AI Configuration")
        
        ai_model = st.selectbox(
            "Primary AI Model",
            ["DeepSeek AI", "GPT-4", "Claude", "Hybrid Ensemble"],
            index=0
        )
        
        ai_temperature = st.slider(
            "AI Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more focused"
        )
        
        st.checkbox("Enable AI Validation", value=True)
        st.checkbox("Enable AI Scoring", value=True)
        st.checkbox("Enable Predictive Analytics", value=True)
        
        if st.button("🔄 Retrain AI Models"):
            st.info("AI models retraining started in background...")
            # This would trigger model retraining
    
    with tab2:
        st.markdown("### 🌍 Country Settings")
        
        # Country configurations
        for country in ["DE", "US", "UK", "UAE", "SA"]:
            with st.expander(f"🇩🇪 {country} Configuration"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_input(f"Email Domains ({country})", 
                                 value="gmail.com, yahoo.com, web.de" if country == "DE" else "gmail.com, yahoo.com")
                with col2:
                    st.text_input(f"Phone Prefix ({country})", 
                                 value="+49" if country == "DE" else "+1" if country == "US" else "+44")
    
    with tab3:
        st.markdown("### 📤 Export Settings")
        
        default_format = st.selectbox(
            "Default Export Format",
            ["CSV", "JSON", "Excel", "Parquet", "Database"],
            index=0
        )
        
        st.checkbox("Include AI Scores in Export", value=True)
        st.checkbox("Include Metadata in Export", value=True)
        st.checkbox("Compress Exports", value=False)
        
        export_path = st.text_input(
            "Export Directory",
            value="./exports"
        )
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")

def generate_sample_leads():
    """Generate sample leads for demonstration"""
    import random
    from datetime import datetime
    
    sample_leads = []
    countries = ["US", "DE", "UK", "CA", "AU", "UAE"]
    first_names = ["John", "Emma", "Michael", "Sophia", "David", "Olivia"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
    
    for i in range(10):
        lead = {
            'lead_id': f'SAMPLE_{datetime.now().strftime("%Y%m%d")}_{i:03d}',
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'email': f'{random.choice(first_names).lower()}.{random.choice(last_names).lower()}@example.com',
            'phone': f'+1{random.randint(200,999)}{random.randint(100,999)}{random.randint(1000,9999)}',
            'country': random.choice(countries),
            'city': random.choice(['New York', 'Berlin', 'London', 'Toronto', 'Sydney', 'Dubai']),
            'lead_type': random.choice(['forex', 'crypto', 'casino']),
            'lead_category': random.choice(['hot', 'recovery', 'xdepositor']),
            'ai_score': round(random.uniform(0.6, 0.95), 2),
            'status': 'new',
            'generated_date': datetime.now().isoformat()
        }
        sample_leads.append(lead)
    
    return sample_leads

def save_to_csv():
    """Save leads to CSV"""
    if st.session_state.leads:
        df = pd.DataFrame(st.session_state.leads)
        filename = f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        # Provide download link
        with open(filename, "rb") as file:
            st.download_button(
                label="📥 Download CSV",
                data=file,
                file_name=filename,
                mime="text/csv"
            )

def save_to_json():
    """Save leads to JSON"""
    if st.session_state.leads:
        filename = f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.leads, f, indent=2)
        
        # Provide download link
        with open(filename, "rb") as file:
            st.download_button(
                label="📥 Download JSON",
                data=file,
                file_name=filename,
                mime="application/json"
            )

if __name__ == "__main__":
    main()