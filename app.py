import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 250
CHANNELS = 3
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

st.set_page_config(page_title='Cats vs. Dogs')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cats_vs_dogs_v2.hdf5')
    return model

def welcome():
    st.write("### Cats vs. Dogs Classification")   
    with st.spinner('Loading Model'):
        model = load_model()
    uploaded_file = st.file_uploader("Choose an Image")
    if st.button("Predict"):
        if (uploaded_file is not None) and (allowed_file(uploaded_file.name)):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', width=300)            
            with st.spinner('Predicting'):
                label = upload_predict(image, model)
            if label[0] >= 0.5:
                st.success("### The Image is Dog")
            else:
                st.success("### The Image is Cat")
        else:
            st.error("Something Went Wrong")



def upload_predict(img, model):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
    image = img
    #image sizing
    size = (IMAGE_SIZE, IMAGE_SIZE)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction


if __name__ == '__main__':
    choice = st.sidebar.selectbox("Model/ Author", ["Try the model", "Who am I?"])
    if choice == "Try the model":
        welcome()
    else:
        st.write("## Author Info.")
        st.write("### Ibrahim M. Nasser")
        st.write("Freelance Machine Learning Engineer")
        st.write("[Website](https://ibrahim-nasser.com/)", 
                 "[Blog](https://blog.ibrahim-nasser.com/)", 
                 "[LinkedIn](https://www.linkedin.com/in/ibrahimnasser96/)",
                 "[GitHub](https://github.com/96ibman)",
                 "[Youtube](https://www.youtube.com/channel/UC7N-dy3UbSBHnwwv-vulBAA)",
                 "[Twitter](https://twitter.com/mleng_ibrahim)"
                 )
        st.image("my_picture.jpeg", width=350)