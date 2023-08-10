import ultralytics
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import gdown

def engine(im):

    

    url = 'https://drive.google.com/uc?id=1g6LwjT6pw0ZkyIdP-jFEBGYwpILDTwjA'
    output = "trained_model.pt"
    gdown.download(url, output, quiet=False)
    
    model = YOLO("trained_model.pt")
    results = model(im)  # results list

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image

def main():
    st.write("Testing")

    from io import StringIO

    user_im = st.file_uploader("Choose a file")
    im = Image.open(user_im).convert('RGB').save('user_image.jpeg')
    
    
    if user_im is not None:
        st.image(user_im)
        
        

    if st.button("Start the Prediction!"):
        engine("user_image.jpeg")

        with open("results.jpg", "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="prediction_image.jpg",
                    mime="image/jpg"
                )
        
    
if __name__ == "__main__":
    main()
