import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Set page config
st.set_page_config(page_title="Animal Classifier", page_icon="🐾", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploaded-image {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained ResNet18 model
@st.cache_resource
def load_model():
    # Load the pre-trained ResNet18
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: cat, dog, panda
    
    # Load your trained weights here
    # model.load_state_dict(torch.load('path_to_your_trained_model.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['cat', 'dog', 'panda']

def predict(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
    return predicted.item(), probabilities[0].tolist()

def main():
    st.title("🐱 Cat vs Dog vs Panda Classifier 🐶🐼")
    st.write("Upload an image of a cat, dog, or panda, and our AI will predict which one it is!")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This is an animal classifier using a pre-trained ResNet18 model with PyTorch. "
        "It can identify between cats, dogs, and pandas."
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', width=256)
        
        # Make prediction
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                # Load model
                model = load_model()
                
                # Make prediction
                predicted_class, probabilities = predict(image, model)
                
                # Display results
                st.markdown("### Prediction Results")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Class Probabilities:**")
                    for i, prob in enumerate(probabilities):
                        emoji = "🐱" if i == 0 else "🐶" if i == 1 else "🐼"
                        st.write(f"{emoji} {class_names[i].title()}: {prob*100:.2f}%")
                
                with col2:
                    emoji = "🐱" if predicted_class == 0 else "🐶" if predicted_class == 1 else "🐼"
                    result = f"{emoji} It's a {class_names[predicted_class].title()}!"
                    st.markdown(f"**Prediction:** {result}")
                    
                    # Show confidence level
                    confidence = max(probabilities) * 100
                    st.progress(int(confidence))
                    st.write(f"Confidence: {confidence:.2f}%")
                
                # Add some spacing
                st.markdown("---")
                
                # Add some tips
                st.markdown("### Tips for Better Results")
                st.markdown("""
                - Use clear, well-lit photos
                - Make sure the animal is clearly visible
                - Try to get a front-facing view
                - Avoid images with multiple animals
                - Works best with close-up shots
                """)

if __name__ == "__main__":
    main()