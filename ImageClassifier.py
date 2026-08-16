import streamlit as st 
import pandas as pd
import numpy as np
import tensorflow as tf
from io import BytesIO
import PIL.Image as Image
import tensorflow as tf

html_temp = """
    <div style ="background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">X-Ray Image Classifier</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
img_size = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

model = tf.keras.models.load_model(r'C:\Users\agyek\Downloads\streamlit_application\pre_trained_model_10.keras')
print('Model Loaded')

def load_classifier():
    st.subheader("Upload an X-Ray image to detect if it is Normal or Pneumonia")
    file = st.file_uploader(label="", type=['jpeg'])

    if file!=None:
        # Convert uploaded file in_memory Image object
        img = Image.open(BytesIO(file.read()))
        img = img.resize((img_size, img_size))
        new_array = tf.keras.preprocessing.image.img_to_array(img)
        new_array = new_array.reshape(-1,img_size,img_size,3)
        st.image(file)
        st.write("")
        st.write("")
        
        if st.button("predict"):
            #Making prediction
            preds = ""
            prediction=model.predict(new_array/255.0)
            print(prediction)
            pred_value = prediction[0][0]
            print(round(pred_value))
            preds = CATEGORIES[int(round(pred_value))] + "-" + str(round(pred_value))
            
def main():
    
    load_classifier()
    
if __name__ == "__main__":
    main()