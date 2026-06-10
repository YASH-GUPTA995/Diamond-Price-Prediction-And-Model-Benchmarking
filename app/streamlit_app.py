import streamlit as st
import pandas as pd
import numpy as np
from price_predictor import train_price_predictor, predict_price

# Page configuration
st.set_page_config(
    page_title="Diamond Price Checker",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2C3E50;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #34495E;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #E8F4F8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
        margin: 10px 0;
    }
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .feature-label {
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 5px;
    }
    .stats-container {
        background-color: #F0F0F0;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for model
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Header
st.markdown('<div class="main-header">💎 DIAMOND PRICE CHECKER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Diamond Price Prediction</div>', unsafe_allow_html=True)

# Load data and train model
@st.cache_resource
def load_model():
    # Use relative path to avoid escaping issues
    df = pd.read_csv('diamonds.csv')
    train_price_predictor(df)
    return df

df = load_model()

# Sidebar with information
with st.sidebar:
    st.header("📊 Dataset Information")
    st.info("""
    **Diamond Dataset Statistics:**
    - Total Diamonds: 53,940
    - Analyzed: 46,434 (after outlier removal)
    - Model Type: XGBoost Regressor
    - R² Score: 0.9809 (98.09% accurate)
    - Mean Error: $196.03
    """)
    
    st.divider()
    
    st.header("📈 Feature Statistics")
    
    # Display statistics for numeric features
    numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    selected_stat = st.selectbox("Select a feature to see statistics:", numeric_features)
    
    if selected_stat in df.columns:
        st.metric("Mean", f"{df[selected_stat].mean():.2f}")
        st.metric("Median", f"{df[selected_stat].median():.2f}")
        st.metric("Min", f"{df[selected_stat].min():.2f}")
        st.metric("Max", f"{df[selected_stat].max():.2f}")

# Main content area
st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("✨ Diamond Features")
    
    # Carat input
    carat = st.slider(
        "Carat (Weight)",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Weight of the diamond in carats. Range: 0.2-5.0"
    )
    
    # Cut selection
    cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut = st.selectbox(
        "Cut Quality",
        cut_options,
        index=2,
        help="Quality of the diamond cut"
    )
    
    # Color selection
    color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    color = st.selectbox(
        "Color Grade",
        color_options,
        index=4,
        help="Color grade from J (lowest) to D (highest)"
    )
    
    # Clarity selection
    clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity = st.selectbox(
        "Clarity Grade",
        clarity_options,
        index=4,
        help="Clarity grade from I1 (lowest) to IF (highest)"
    )

with col2:
    st.subheader("📏 Physical Dimensions")
    
    # Depth
    depth = st.slider(
        "Depth %",
        min_value=43.0,
        max_value=79.0,
        value=62.0,
        step=0.1,
        help="Total depth percentage (43-79)"
    )
    
    # Table
    table = st.slider(
        "Table %",
        min_value=43.0,
        max_value=95.0,
        value=57.0,
        step=0.1,
        help="Width of top of diamond relative to widest point"
    )
    
    # X dimension
    x = st.slider(
        "Length (X) - mm",
        min_value=3.0,
        max_value=11.0,
        value=5.5,
        step=0.01,
        help="Length of diamond in millimeters"
    )
    
    # Y dimension
    y = st.slider(
        "Width (Y) - mm",
        min_value=3.0,
        max_value=11.0,
        value=5.5,
        step=0.01,
        help="Width of diamond in millimeters"
    )
    
    # Z dimension
    z = st.slider(
        "Depth (Z) - mm",
        min_value=2.0,
        max_value=6.0,
        value=3.3,
        step=0.01,
        help="Depth of diamond in millimeters"
    )

st.markdown("---")

# Display current diamond specifications
st.subheader("💍 Selected Diamond Specifications")

spec_col1, spec_col2, spec_col3 = st.columns(3)

with spec_col1:
    st.markdown(f"""
    <div class="info-box">
    <div class="feature-label">Carat Weight</div>
    <div style="font-size: 1.5em; color: #667eea;">{carat:.2f} ct</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <div class="feature-label">Cut Quality</div>
    <div style="font-size: 1.5em; color: #667eea;">{cut}</div>
    </div>
    """, unsafe_allow_html=True)

with spec_col2:
    st.markdown(f"""
    <div class="info-box">
    <div class="feature-label">Color Grade</div>
    <div style="font-size: 1.5em; color: #667eea;">{color}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <div class="feature-label">Clarity Grade</div>
    <div style="font-size: 1.5em; color: #667eea;">{clarity}</div>
    </div>
    """, unsafe_allow_html=True)

with spec_col3:
    st.markdown(f"""
    <div class="info-box">
    <div class="feature-label">Depth %</div>
    <div style="font-size: 1.5em; color: #667eea;">{depth:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <div class="feature-label">Table %</div>
    <div style="font-size: 1.5em; color: #667eea;">{table:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Predict button
predict_button = st.button(
    "🔮 PREDICT PRICE",
    use_container_width=True,
    type="primary"
)

if predict_button:
    # Create features dictionary
    features = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    
    try:
        # Make prediction
        predicted_price = predict_price(features)
        
        # Display predicted price in a highlighted box
        st.markdown(f"""
        <div class="price-box">
        💰 Predicted Price<br>
        <span style="font-size: 0.8em; opacity: 0.9;">USD</span><br>
        ${predicted_price:,.2f}
        </div>
        """, unsafe_allow_html=True)
        
        # Additional information
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            # Price per carat
            price_per_carat = predicted_price / carat
            st.metric("Price per Carat", f"${price_per_carat:,.2f}")
        
        with col_info2:
            # Diamond size indicator
            if carat < 0.5:
                size_desc = "Modest"
            elif carat < 1.0:
                size_desc = "Small-Medium"
            elif carat < 2.0:
                size_desc = "Large"
            else:
                size_desc = "Very Large"
            st.metric("Diamond Size", size_desc)
        
        with col_info3:
            # Quality indicator
            quality_score = 0
            if cut in ['Premium', 'Ideal']:
                quality_score += 2
            else:
                quality_score += 1
            
            if color in ['D', 'E', 'F']:
                quality_score += 2
            elif color in ['G', 'H']:
                quality_score += 1
            
            if clarity in ['VVS1', 'VVS2', 'IF']:
                quality_score += 2
            elif clarity in ['VS1', 'VS2']:
                quality_score += 1
            
            quality_stars = "⭐" * min(quality_score, 5)
            st.metric("Quality Rating", quality_stars)
        
        # Detailed breakdown
        st.divider()
        st.subheader("📋 Prediction Details")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write("**Input Features:**")
            input_df = pd.DataFrame({
                'Feature': ['Carat', 'Cut', 'Color', 'Clarity', 'Depth', 'Table', 'X', 'Y', 'Z'],
                'Value': [f'{carat}', cut, color, clarity, f'{depth:.2f}%', f'{table:.2f}%', f'{x:.2f}mm', f'{y:.2f}mm', f'{z:.2f}mm']
            })
            st.table(input_df)
        
        with detail_col2:
            st.write("**Model Information:**")
            info_df = pd.DataFrame({
                'Metric': ['Model Type', 'R² Score', 'Mean Error', 'Training Data', 'Status'],
                'Value': ['XGBoost Regressor', '0.9809', '$196.03', '46,434 diamonds', '✅ Active']
            })
            st.table(info_df)
    
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        st.info("Please ensure all input values are within valid ranges.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #7F8C8D; font-size: 0.9em; margin-top: 30px;">
    <p>💎 Diamond Price Checker | Powered by XGBoost Machine Learning Model</p>
    <p>Model Accuracy: R² = 0.9809 | Mean Absolute Error: $196.03</p>
    <p>© 2025 Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
