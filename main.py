import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from model import NeuralNetwork
st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(14, 165, 233, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(168, 85, 247, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #111827
        );

    min-height: 100vh;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.main-title {

    text-align: center;

    font-size: 52px;

    font-weight: 800;

    background: linear-gradient(
        90deg,
        #0ea5e9,
        #6366f1,
        #a855f7
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
    margin-bottom: 5px;
}
.subtitle {

    text-align: center;

    font-size: 18px;

    color: #94a3b8;

    margin-bottom: 35px;
}
.section-card {

    background: rgba(
        15,
        23,
        42,
        0.75
    );

    border: 1px solid
    rgba(
        148,
        163,
        184,
        0.15
    );

    border-radius: 20px;

    padding: 25px;

    margin-top: 15px;

    backdrop-filter: blur(10px);
}

.result-card {

    padding: 35px;

    border-radius: 20px;

    background: linear-gradient(
        135deg,
        rgba(30, 41, 59, 0.90),
        rgba(49, 46, 129, 0.40)
    );

    border: 1px solid
    rgba(
        129,
        140,
        248,
        0.40
    );

    text-align: center;

    margin-top: 15px;

    margin-bottom: 25px;

    box-shadow:
        0px 10px 40px
        rgba(0, 0, 0, 0.25);
}

.prediction-title {

    font-size: 14px;

    color: #94a3b8;

    letter-spacing: 3px;

    margin-bottom: 10px;
}
.prediction {

    font-size: 100px;

    line-height: 1.1;

    font-weight: 900;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    margin-top: 10px;

    margin-bottom: 10px;
}
.confidence {

    font-size: 21px;

    color: #cbd5e1;
}


.confidence-number {

    color: #38bdf8;

    font-weight: 700;
}
.stButton > button {

    width: 100%;

    height: 52px;

    border-radius: 12px;

    border: 1px solid
    rgba(
        129,
        140,
        248,
        0.50
    );

    background: linear-gradient(
        90deg,
        #0284c7,
        #7c3aed
    );

    color: white;

    font-size: 17px;

    font-weight: 600;

    transition: 0.3s;
}


.stButton > button:hover {

    transform: translateY(-2px);

    border-color: #38bdf8;

    color: white;

    box-shadow:
        0px 8px 25px
        rgba(
            56,
            189,
            248,
            0.25
        );
}
button[data-baseweb="tab"] {

    font-size: 17px;

    font-weight: 600;

    padding-left: 25px;

    padding-right: 25px;
}
[data-testid="stFileUploader"] {

    background: rgba(
        15,
        23,
        42,
        0.40
    );

    border-radius: 15px;

    padding: 10px;
}


h1, h2, h3 {

    color: #e2e8f0;
}


p {

    color: #94a3b8;
}



.footer {

    text-align: center;

    color: #64748b;

    margin-top: 60px;

    margin-bottom: 20px;

    font-size: 14px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


@st.cache_resource
def load_model():

    model = NeuralNetwork()

    model.load_state_dict(

        torch.load(

            "mnist_model.pth",

            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model


model = load_model()


def preprocess_image(image):

    image = image.convert("L")
    image_array = np.array(image)


    if np.mean(image_array) > 127:

        image = ImageOps.invert(image)

    image = image.resize((28, 28))


    transform = transforms.ToTensor()


    image_tensor = transform(image)


    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    return image_tensor

def predict_digit(image):


    # Preprocess image

    image_tensor = preprocess_image(image)


    # Turn off gradient calculation

    with torch.no_grad():


        # Forward propagation

        output = model(image_tensor)


        # Convert logits into probabilities

        probabilities = F.softmax(

            output,

            dim=1
        )


        # Find predicted digit

        prediction = torch.argmax(

            probabilities,

            dim=1

        ).item()


        # Get confidence

        confidence = probabilities[

            0,

            prediction

        ].item()


    # Convert probabilities to NumPy

    probabilities = (

        probabilities

        .cpu()

        .numpy()[0]
    )


    return (

        prediction,

        confidence,

        probabilities
    )


# ============================================================
# RESULT DISPLAY FUNCTION
# ============================================================

def display_result(

    prediction,

    confidence,

    probabilities
):


    # Prediction Card

    st.markdown(

        f'<div class="result-card">'
        f'<div class="prediction-title">'
        f'MODEL PREDICTION'
        f'</div>'
        f'<div class="prediction">'
        f'{prediction}'
        f'</div>'
        f'<div class="confidence">'
        f'Confidence: '
        f'<span class="confidence-number">'
        f'{confidence * 100:.2f}%'
        f'</span>'
        f'</div>'
        f'</div>',

        unsafe_allow_html=True
    )


    # Probability heading

    st.subheader(
        "📊 Prediction Probabilities"
    )


    # Create dataframe

    probability_data = pd.DataFrame({

        "Digit": [
            str(i)
            for i in range(10)
        ],

        "Probability":

            probabilities

    })


    # Display chart

    st.bar_chart(

        probability_data,

        x="Digit",

        y="Probability",

        height=350
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(

    '<div class="main-title">'
    '🧠 AI Digit Recognizer'
    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# SUBTITLE
# ============================================================

st.markdown(

    '<div class="subtitle">'
    'Draw or upload a handwritten digit and '
    'let the neural network recognize it.'
    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

draw_tab, upload_tab = st.tabs(

    [

        "✏️ Draw Digit",

        "📤 Upload Image"

    ]
)


# ============================================================
# DRAW DIGIT TAB
# ============================================================

with draw_tab:


    # Create two columns

    left_column, right_column = st.columns(

        [1, 1.2],

        gap="large"
    )


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left_column:


        st.subheader(

            "✏️ Draw a Digit"

        )


        st.write(

            "Draw any digit from 0 to 9 "
            "inside the canvas."

        )


        # Drawing canvas

        canvas_result = st_canvas(

            fill_color="black",

            stroke_width=20,

            stroke_color="white",

            background_color="black",

            height=350,

            width=350,

            drawing_mode="freedraw",

            return_image_data=True,

            key="drawing_canvas"
        )


        # Predict button

        predict_draw = st.button(

            "✨ Predict Drawn Digit",

            key="predict_draw",

            use_container_width=True
        )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right_column:


        if predict_draw:


            if canvas_result.image_data is not None:


                # Get canvas image

                image_data = (

                    canvas_result.image_data

                )


                # Check if canvas is empty

                grayscale_check = np.mean(

                    image_data[:, :, :3]

                )


                if grayscale_check < 1:


                    st.warning(

                        "Please draw a digit first."

                    )


                else:


                    # Convert canvas data to image

                    image = Image.fromarray(

                        image_data.astype(

                            "uint8"

                        )
                    )


                    # Make prediction

                    (

                        prediction,

                        confidence,

                        probabilities

                    ) = predict_digit(image)


                    # Display result

                    display_result(

                        prediction,

                        confidence,

                        probabilities
                    )


        else:


            # Default information

            st.markdown(

                '<div class="result-card">'
                '<div class="prediction-title">'
                'MODEL PREDICTION'
                '</div>'
                '<div class="prediction">'
                '?'
                '</div>'
                '<div class="confidence">'
                'Draw a digit and click Predict'
                '</div>'
                '</div>',

                unsafe_allow_html=True
            )


# ============================================================
# UPLOAD IMAGE TAB
# ============================================================

with upload_tab:


    # Two columns

    left_column, right_column = st.columns(

        [1, 1.2],

        gap="large"
    )


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left_column:


        st.subheader(

            "📤 Upload Digit Image"

        )


        st.write(

            "Upload an image containing "
            "a handwritten digit."

        )


        # File uploader

        uploaded_file = st.file_uploader(

            "Choose PNG, JPG or JPEG image",

            type=[

                "png",

                "jpg",

                "jpeg"

            ]
        )


        if uploaded_file is not None:


            # Open image

            uploaded_image = Image.open(

                uploaded_file

            )


            # Display uploaded image

            st.image(

                uploaded_image,

                caption="Uploaded Image",

                width=350
            )


            # Predict button

            predict_upload = st.button(

                "✨ Predict Uploaded Image",

                key="predict_upload",

                use_container_width=True
            )


        else:

            predict_upload = False


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right_column:


        if predict_upload:


            # Make prediction

            (

                prediction,

                confidence,

                probabilities

            ) = predict_digit(

                uploaded_image

            )


            # Display result

            display_result(

                prediction,

                confidence,

                probabilities
            )


        else:


            # Default result card

            st.markdown(

                '<div class="result-card">'
                '<div class="prediction-title">'
                'MODEL PREDICTION'
                '</div>'
                '<div class="prediction">'
                '?'
                '</div>'
                '<div class="confidence">'
                'Upload an image and click Predict'
                '</div>'
                '</div>',

                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(

    '<div class="footer">'
    'Built with ❤️ using PyTorch & Streamlit'
    '</div>',

    unsafe_allow_html=True
)