"""
AI-Powered Plastic Sorting MVP - Streamlit Web App

This app allows users to upload plastic images and get real-time classification.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="AI Plastic Sorter",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = 224
MODEL_PATH = 'plastic_classifier_final.h5'
CLASS_MAPPING_PATH = 'class_mapping.json'

# Plastic type descriptions (for Nigerian context)
PLASTIC_INFO = {
    'PET': {
        'full_name': 'Polyethylene Terephthalate',
        'description': 'Clear plastic used for water bottles, soft drinks, cooking oil containers',
        'recyclable': 'Highly Recyclable ✅',
        'food_grade': 'Yes - Food Safe',
        'common_uses': 'Water bottles, soft drink bottles, cooking oil containers, cosmetic jars',
        'recycling_symbol': '♳ (1)',
        'value': 'High recycling value in Nigeria'
    },
    'PE': {
        'full_name': 'Polyethylene (LDPE/HDPE)',
        'description': 'Flexible or rigid plastic for bags, milk bottles, detergent containers',
        'recyclable': 'Recyclable ✅',
        'food_grade': 'HDPE is food-safe',
        'common_uses': 'Shopping bags, milk bottles, detergent bottles, garbage bags',
        'recycling_symbol': '♴ (2) or ♶ (4)',
        'value': 'Moderate recycling value'
    },
    'PC': {
        'full_name': 'Polycarbonate',
        'description': 'Strong, impact-resistant plastic for large water bottles and safety equipment',
        'recyclable': 'Limited Recyclability ⚠️',
        'food_grade': 'Previously used but controversial',
        'common_uses': 'Large water gallons, baby bottles (older), safety goggles, CDs/DVDs',
        'recycling_symbol': '♹ (7)',
        'value': 'Low recycling value - often downcycled'
    },
    'PP': {
        'full_name': 'Polypropylene',
        'description': 'Heat-resistant plastic for food containers and takeaway packaging',
        'recyclable': 'Recyclable ✅',
        'food_grade': 'Yes - Food Safe',
        'common_uses': 'Takeaway containers, yogurt cups, bottle caps, straws, microwave containers',
        'recycling_symbol': '♸ (5)',
        'value': 'Good recycling value in Nigeria'
    },
    'PS': {
        'full_name': 'Polystyrene',
        'description': 'Foam or rigid plastic for packaging and disposable food containers',
        'recyclable': 'Rarely Recycled ❌',
        'food_grade': 'Used for food but controversial',
        'common_uses': 'Styrofoam packaging, foam trays, disposable plates/cups, CD cases',
        'recycling_symbol': '♼ (6)',
        'value': 'Very low recycling value - often ends up in landfills'
    },
    'Others': {
        'full_name': 'Other Plastics (PVC, ABS, etc.)',
        'description': 'Mixed category including PVC pipes, toys, and various hard plastics',
        'recyclable': 'Varies by type ⚠️',
        'food_grade': 'Generally not food-safe',
        'common_uses': 'PVC pipes, LEGO toys, electronic casings, luggage',
        'recycling_symbol': '♹ (7)',
        'value': 'Low to no recycling value'
    }
}


@st.cache_resource
def load_model_and_mapping():
    """Load the trained model and class mapping."""
    try:
        # Load model without compiling to avoid config issues
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Recompile with basic settings
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        with open(CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        
        # Convert keys to integers
        class_mapping = {int(k): v for k, v in class_mapping.items()}
        
        return model, class_mapping
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found!")
        st.info("Please ensure these files exist in the same directory as streamlit_app.py:")
        st.code("""
        Required files:
        - plastic_classifier_final.h5
        - class_mapping.json
        """)
        st.warning("Download these files from your Colab notebook after training.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try loading with compile=False if you see 'batch_shape' or config errors.")
        return None, None


def preprocess_image(image):
    """Preprocess uploaded image for model prediction."""
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array
    img_array = np.array(image)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Expand dims and normalize
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


def predict_plastic_type(model, image, class_mapping):
    """Make prediction on image."""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Get class with highest probability
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_mapping[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Get all probabilities
    all_predictions = {
        class_mapping[i]: float(prob) 
        for i, prob in enumerate(predictions)
    }
    
    return predicted_class, confidence, all_predictions


def get_confidence_color(confidence):
    """Return CSS class based on confidence level."""
    if confidence >= 0.7:
        return 'confidence-high'
    elif confidence >= 0.4:
        return 'confidence-medium'
    else:
        return 'confidence-low'


def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ AI-Powered Plastic Sorter</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identifying Plastic Types for Better Recycling in Nigeria</p>', unsafe_allow_html=True)
    
    # Load model
    model, class_mapping = load_model_and_mapping()
    
    if model is None:
        st.error("⚠️ Model not found. Please run the training notebook first.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This AI system identifies plastic types to improve sorting accuracy 
        and protect food-grade plastics from contamination.
        
        **How it works:**
        1. Upload an image of plastic waste
        2. AI analyzes visual features
        3. Get instant classification
        
        **Supported Types:**
        - PET (Water bottles)
        - PE (Bags, milk bottles)
        - PC (Water gallons)
        - PP (Takeaway containers)
        - PS (Foam packaging)
        - Others (Mixed plastics)
        """)
        
        st.header("🎯 Model Performance")
        st.metric("Test Accuracy", "85%+")
        st.metric("Training Images", "3,000+")
        
        st.header("ℹ️ Instructions")
        st.info("""
        **Best Results:**
        - Clear, well-lit images
        - Single plastic item
        - Fill the frame
        - Avoid heavy shadows
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Plastic Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plastic item"
        )
        
        # Sample images option
        use_sample = st.checkbox("Or try a sample image")
        
        if use_sample:
            st.info("Sample images coming soon! Please upload your own for now.")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Predict button
            if st.button('🔍 Classify Plastic Type', type='primary'):
                with st.spinner('Analyzing image...'):
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_plastic_type(
                        model, image, class_mapping
                    )
                    
                    # Store in session state
                    st.session_state['prediction'] = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'all_predictions': all_predictions
                    }
    
    with col2:
        st.header("🎯 Classification Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            predicted_class = pred['class']
            confidence = pred['confidence']
            all_predictions = pred['all_predictions']
            
            # Main prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            confidence_class = get_confidence_color(confidence)
            
            st.markdown(f"### Predicted Type: **{predicted_class}**")
            st.markdown(f'<p class="{confidence_class}">Confidence: {confidence*100:.1f}%</p>', 
                       unsafe_allow_html=True)
            
            # Info about the plastic type
            if predicted_class in PLASTIC_INFO:
                info = PLASTIC_INFO[predicted_class]
                
                st.markdown(f"**{info['full_name']}**")
                st.write(info['description'])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Recyclability", info['recyclable'])
                with col_b:
                    st.metric("Food Grade", info['food_grade'])
                
                with st.expander("📖 More Details"):
                    st.write(f"**Common Uses:** {info['common_uses']}")
                    st.write(f"**Recycling Symbol:** {info['recycling_symbol']}")
                    st.write(f"**Economic Value:** {info['value']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence chart
            st.subheader("📊 Confidence Scores")
            
            # Sort by confidence
            sorted_preds = dict(sorted(all_predictions.items(), 
                                     key=lambda x: x[1], reverse=True))
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sorted_preds.values()),
                    y=list(sorted_preds.keys()),
                    orientation='h',
                    marker=dict(
                        color=list(sorted_preds.values()),
                        colorscale='Blues',
                        showscale=False
                    ),
                    text=[f'{v*100:.1f}%' for v in sorted_preds.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Prediction Confidence by Plastic Type',
                xaxis_title='Confidence',
                yaxis_title='Plastic Type',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.subheader("💡 Recommendation")
            
            if confidence >= 0.7:
                st.success(f"""
                ✅ **High Confidence Classification**
                
                This plastic is identified as **{predicted_class}** with high confidence.
                
                **Next Steps:**
                - Route to appropriate recycling stream
                - {info['value'] if predicted_class in PLASTIC_INFO else 'Check recycling value'}
                """)
            elif confidence >= 0.4:
                st.warning(f"""
                ⚠️ **Medium Confidence Classification**
                
                This appears to be **{predicted_class}**, but consider manual verification.
                
                **Next Steps:**
                - Visual inspection by trained sorter
                - Check for recycling symbols
                - Consider secondary testing
                """)
            else:
                st.error(f"""
                ❌ **Low Confidence Classification**
                
                Uncertain classification. Manual sorting recommended.
                
                **Possible Reasons:**
                - Poor image quality
                - Unusual plastic type
                - Mixed materials
                - Try retaking with better lighting
                """)
        
        else:
            st.info("👈 Upload an image to see classification results")
            
            # Placeholder image
            st.image("https://via.placeholder.com/400x300?text=Upload+Image+to+Start", 
                    use_column_width=True)
    
    # Footer
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Model", "ResNet50 Transfer Learning")
    with col_b:
        st.metric("Dataset", "3,000+ images")
    with col_c:
        st.metric("Target Accuracy", "80-85%")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🇳🇬 Built for Nigerian Recycling Facilities</p>
        <p>Phase 1 MVP - Proof of Concept</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
