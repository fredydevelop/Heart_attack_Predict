#Importing the dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
#from sklearn.model_selection import train_test_split

import streamlit as st
import base64
import pickle as pk
#import joblib





#configuring the page setup
st.set_page_config(page_title='Heart attack prediction system',layout='centered')

#selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)
with st.sidebar:
    st.title("Home Page")
    selection=st.radio("select your option",options=["Predict for a Single-Patient", "Predict for Multi-Patient"])


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def Heart_Attack(givendata):
    # loaded_model = joblib.load("heartcheck.sav")

    loaded_model=pk.load(open("heartchecks.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    if prediction==1:
      return "Heart Issues present"
    else:
      return "No Heart Issues Present"
    

#main function handling the input
def main():
    st.header("Heart Attack Detection and Predictive System")
    
    #getting user input
    
    age = st.slider('Patient age', 0, 200, key="ageslide")
    st.write("Patient's is :", age, 'years old')

    option1 = st.selectbox('sex',("",'Male' ,'Female'),key="gender")
    if (option1=='Male'):
        sex=1
    else:
        sex=0

    option4 = st.selectbox('Chest Pain type',("","typical angina","atypical angina","non-anginal pain","asymptomatic"),key="cps")
    if (option4=="typical angina"):
        cp=0
    elif (option4=='atypical angina'):
        cp=1
    elif (option4=='non-anginal pain'):
        cp=2
    else:
        cp=3

    option5 = st.slider('resting blood pressure (in mm Hg)',100,300,key="trtbps")
    st.write("Patient esting blood pressure (in mm Hg) is: ", option5)


    option14 = st.slider("cholestoral in mg/dl fetched via BMI sensor",100,1200,key="chol")
    st.write("Your cholesterol level  is: ", option14)


    option6 = st.selectbox('fasting blood sugar > 120 mg/dl)',("",'True' ,'False'),key="fbst")
    if (option6=='True'):
        fbs=1
    else:
        fbs=0


    option7 = st.selectbox('resting electrocardiographic results',("",'normal' ,'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)',"showing probable or definite left ventricular hypertrophy by Estes' criteria"),key="restecgs")
    if (option7=='normal'):
        restecg=0
    elif (option7=='having ST-T wave abnormality' ):
        restecg=1
    else:
        restecg=2


    option8 = st.slider('Maximum heart rate achieved',10, 400,key="thalachh")
    st.write("Patient's maximum heart rate is: ", option8)


    option2 = st.selectbox('exercise induced angina',("",'Yes' ,'No'),key="exng")
    if (option2=='Yes'):
        exang=1
    else:
        exang=0

    
    option9 = st.number_input('Insert Previous Number',key="oldPeak")
    st.write('The current number is ', option9)

    option15 = st.slider('slope',0,2,key="slope")
    st.write("Patient slope is: ", option15)

    option3 = st.selectbox('number of major vessels',("","0","1" ,"2","3"),key="caas")
    if (option3=='0'):
        caa=0

    elif (option3=="1"):
        caa=1

    elif (option3=='2'):
        caa=2
    else:
        caa=3
    

    option10 = st.slider('thal rate',0,3,key="thall")
    st.write('The thal rate is ', option10)



    st.write("\n")
    st.write("\n")





    detectionResult = ''#for displaying result
    
    # creating a button for Prediction
    if age!="" and option1!=""  and option2!=""  and option3!=""  and option4!="" and option5!="" and option6!="" and option7 !=""and  option8 !="" and option9!="" and option10 !="" and option14 !="" and option15!="" and st.button('Predict'):
        detectionResult = Heart_Attack([age,sex,cp,option5,option14, fbs,restecg, option8, exang, option9, option15,caa, option10])
        st.success(detectionResult)


def multi(input_data):
    loaded_model=pk.load(open("heartchecks.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    if "output" in dfinput.iloc[1:]:
        dfinput.drop("output",axis=1,inplace=True)
    dfinput=dfinput.reset_index(drop=True)

    st.header('A view of the uploaded dataset')
    st.markdown('')
    st.dataframe(dfinput)

    dfinput=dfinput.values
    std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
    std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
    predict=st.button("predict")


    if predict:
        prediction = loaded_model.predict(std_dfinput)
        interchange=[]
        for i in prediction:
            if i==1:
                newi="Heart issues present"
                interchange.append(newi)
            elif i==0:
                newi="No heart issues present"
                interchange.append(newi)
            
        st.subheader('All the predictions')
        prediction_output = pd.Series(interchange, name='Heart attack prediction results')
        prediction_id = pd.Series(np.arange(len(interchange)),name="Patient_ID")
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

if selection =="Predict for a Single-Patient":
    main()

if selection == "Predict for Multi-Patient":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv","xls"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    
